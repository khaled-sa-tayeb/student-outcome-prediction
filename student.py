import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import GroupKFold, cross_val_predict

from features import KEYS, NUMERIC, encode

# One model per checkpoint: each only sees data available by that course day,
# so an early checkpoint gives an early warning and later ones are more accurate.
CHECKPOINT_DAYS = [30, 60, 90, 120, 150]

# Candidate decision thresholds; the one with the best cross-validated F1 is used
THRESHOLDS = [0.45, 0.5, 0.55, 0.6, 0.65]

# Share of students held out for testing (the same students at every checkpoint)
TEST_SHARE = 0.2

# ============================= 1. Load Data =============================
student_info = pd.read_csv("studentInfo.csv")
student_registration = pd.read_csv("studentRegistration.csv")
student_assessment = pd.read_csv("studentAssessment.csv")
assessments = pd.read_csv("assessments.csv")
student_vle = pd.read_csv("studentVle.csv", usecols=KEYS + ["date", "sum_click"])

print("VLE activity records:", len(student_vle))

# Dataset quirk: one band is stored without the % sign
student_info["imd_band"] = student_info["imd_band"].replace("10-20", "10-20%")

# Attach module/presentation/deadline to each submission
submissions_all = student_assessment.merge(
    assessments[["id_assessment", "code_module", "code_presentation", "date"]], on="id_assessment"
)
submissions_all["days_late"] = submissions_all["date_submitted"] - submissions_all["date"]


# ============================= 2. Feature Engineering =============================
def build_features(cutoff_day):
    """One row per enrollment (student + module + presentation) using data up to cutoff_day."""
    vle = student_vle[student_vle["date"] <= cutoff_day]

    # --- Engagement on the VLE ---
    vle_groups = vle.groupby(KEYS)
    engagement = pd.DataFrame({
        "total_clicks": vle_groups["sum_click"].sum(),
        "active_days": vle_groups["date"].nunique(),
        "last_active_day": vle_groups["date"].max(),
    }).reset_index()
    engagement["clicks_per_active_day"] = engagement["total_clicks"] / engagement["active_days"]
    engagement["days_since_last_activity"] = cutoff_day - engagement.pop("last_active_day")

    # --- Recent activity: clicks in the two weeks before the cutoff ---
    recent = vle[vle["date"] > cutoff_day - 14].groupby(KEYS)["sum_click"].sum()
    engagement = engagement.merge(recent.rename("clicks_last_14_days").reset_index(), on=KEYS, how="left")

    # --- Assessment results ---
    submissions = submissions_all[submissions_all["date_submitted"] <= cutoff_day]
    assessment_groups = submissions.groupby(KEYS)
    performance = pd.DataFrame({
        "avg_score": assessment_groups["score"].mean(),
        "num_assessments": assessment_groups["id_assessment"].count(),
        "avg_days_late": assessment_groups["days_late"].mean(),
    }).reset_index()

    # --- Missed deadlines: assessments due by the cutoff that the student has not submitted ---
    due = assessments[(assessments["date"] <= cutoff_day) & (assessments["assessment_type"] != "Exam")]
    due_per_student = student_info[KEYS].merge(
        due[["code_module", "code_presentation", "id_assessment"]], on=["code_module", "code_presentation"]
    )
    submitted = submissions[["id_student", "id_assessment"]].assign(submitted=1)
    due_per_student = due_per_student.merge(submitted, on=["id_student", "id_assessment"], how="left")
    missed = (
        due_per_student["submitted"].isna()
        .groupby([due_per_student[k] for k in KEYS]).sum()
        .rename("missed_assessments").reset_index()
    )

    # --- Build the enrollments table ---
    data = student_info.merge(student_registration, on=KEYS, how="left")
    for table in [engagement, performance, missed]:
        data = data.merge(table, on=KEYS, how="left")

    fill_zero = ["total_clicks", "active_days", "clicks_per_active_day", "avg_score", "num_assessments",
                 "clicks_last_14_days", "missed_assessments", "avg_days_late"]
    data[fill_zero] = data[fill_zero].fillna(0)
    # No activity at all: treat as inactive for longer than the observed window
    data["days_since_last_activity"] = data["days_since_last_activity"].fillna(cutoff_day + 30)

    # Students who already withdrew by the cutoff need no prediction
    data = data[~(data["date_unregistration"] <= cutoff_day)].reset_index(drop=True)

    # --- Target: Pass/Distinction = 1, Fail/Withdrawn = 0 ---
    data["target"] = data["final_result"].isin(["Pass", "Distinction"]).astype(int)
    return data


def make_model():
    # Settings chosen by 5-fold cross-validation on the training set
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, max_iter=500,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=42,
    )


# ============================= 3. Fixed Test Students =============================
# Chosen once, so every checkpoint is evaluated on the same unseen students
all_students = np.sort(student_info["id_student"].unique())
test_students = set(np.random.RandomState(42).choice(
    all_students, int(len(all_students) * TEST_SHARE), replace=False
))

# ============================= 4. Train One Model per Checkpoint =============================
os.makedirs("Model", exist_ok=True)
summary = []
history = []

for day in CHECKPOINT_DAYS:
    print(f"\n==================== Day {day} ====================")
    data = build_features(day)
    X = encode(data)
    y = data["target"]

    is_test = data["id_student"].isin(test_students).values
    X_train, X_test = X[~is_test], X[is_test]
    y_train, y_test = y[~is_test], y[is_test]

    # --- Pick the decision threshold with cross-validation on the training set only ---
    cv_proba = cross_val_predict(
        make_model(), X_train, y_train, groups=data.loc[~is_test, "id_student"],
        cv=GroupKFold(n_splits=5), method="predict_proba",
    )[:, 1]
    cv_f1 = {t: f1_score(y_train, (cv_proba >= t).astype(int), average="macro") for t in THRESHOLDS}
    threshold = max(cv_f1, key=cv_f1.get)
    cv_accuracy = accuracy_score(y_train, (cv_proba >= threshold).astype(int))

    model = make_model()
    model.fit(X_train, y_train)
    y_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)

    # --- Evaluation ---
    metrics = {
        "cutoff_day": day,
        "enrollments": int(len(data)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "threshold": threshold,
        "cv_accuracy": round(cv_accuracy, 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "at_risk_recall": round(recall_score(y_test, y_pred, pos_label=0), 4),
    }
    print("Training data size:", metrics["train_size"], "| Testing data size:", metrics["test_size"])
    print("Threshold:", threshold, "| CV accuracy:", metrics["cv_accuracy"])
    print("Accuracy:", metrics["accuracy"], "| F1 (macro):", metrics["f1_macro"],
          "| At-risk recall:", metrics["at_risk_recall"])
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # --- Save this checkpoint ---
    folder = f"Model/day_{day}"
    os.makedirs(folder, exist_ok=True)
    joblib.dump(model, f"{folder}/student_model.pkl", compress=3)
    joblib.dump(list(X.columns), f"{folder}/feature_columns.pkl")
    with open(f"{folder}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    summary.append(metrics)
    history.append(data[KEYS + NUMERIC + ["final_result", "target"]].assign(checkpoint_day=day))

# ============================= 5. Save Summary =============================
with open("Model/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n==================== Summary ====================")
print(pd.DataFrame(summary)[["cutoff_day", "accuracy", "f1_macro", "at_risk_recall", "test_size"]].to_string(index=False))

# Values at each checkpoint, used by the app to compare a student with others
pd.concat(history).to_csv("decoded_student_data.csv", index=False)
