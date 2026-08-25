import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================= 1. Load Data =============================
student_info = pd.read_csv("studentInfo.csv")
student_vle = pd.read_csv("studentVle.csv")
student_assessment = pd.read_csv("studentAssessment.csv")
student_registration = pd.read_csv("studentRegistration.csv")
courses = pd.read_csv("courses.csv")
assessments = pd.read_csv("assessments.csv")
vle = pd.read_csv("vle.csv")

# ============================= 2. Feature Engineering =============================

# --- Student Average Score ---
avg_score = student_assessment.groupby("id_student")["score"].mean().reset_index()
avg_score.rename(columns={"score": "avg_score"}, inplace=True)

# --- Number of Assessments Submitted by Student ---
num_assessments = student_assessment.groupby("id_student")["id_assessment"].count().reset_index()
num_assessments.rename(columns={"id_assessment": "num_assessments"}, inplace=True)

# --- Total Clicks ---
total_clicks = student_vle.groupby("id_student")["sum_click"].sum().reset_index()
total_clicks.rename(columns={"sum_click": "total_clicks"}, inplace=True)

# --- Number of Active Days ---
active_days = student_vle.groupby("id_student")["date"].nunique().reset_index()
active_days.rename(columns={"date": "active_days"}, inplace=True)

# --- Average Clicks Per Day ---
avg_clicks_per_day = student_vle.groupby("id_student")["sum_click"].mean().reset_index()
avg_clicks_per_day.rename(columns={"sum_click": "avg_clicks_per_day"}, inplace=True)

# --- Withdrawal Status (using student_registration) ---
student_info = student_info.merge(student_registration[["id_student", "date_unregistration"]], on="id_student", how="left")
student_info["withdrew"] = student_info["date_unregistration"].notnull().astype(int)

# ============================= 3. Build Features Table =============================

# Start from the original data copy (before encoding)
decoded_features = student_info.copy()

features = student_info.copy()

# --- Convert disability column from 'Y'/'N' to 1/0 ---
features["disability"] = features["disability"].map({"Y": 1, "N": 0})

# --- Merge Features ---
features = features.merge(avg_score, on="id_student", how="left")
features = features.merge(num_assessments, on="id_student", how="left")
features = features.merge(total_clicks, on="id_student", how="left")
features = features.merge(active_days, on="id_student", how="left")
features = features.merge(avg_clicks_per_day, on="id_student", how="left")

# --- Target: Map final_result to 0/1 ---
features["target"] = features["final_result"].map(lambda x: 1 if x in ["Pass", "Distinction"] else 0)

# --- Encode Categorical Variables ---
features = pd.get_dummies(features, columns=[
    "gender", 
    "region", 
    "highest_education", 
    "age_band", 
    "imd_band"
], drop_first=True)

# --- Handle Missing Values ---
features.fillna(0, inplace=True)

# ============================= 4. Train Model =============================
X = features.drop(columns=["final_result", "target", "id_student", "code_module", "code_presentation", "date_unregistration"])
y = features["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Training data size:", X_train.shape[0])
print("Testing data size:", X_test.shape[0])
# ============================= 5. Evaluation =============================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import joblib
joblib.dump(model, "student_model.pkl")

#================================================== Save =========================================================

# Add calculated features
decoded_features = decoded_features.merge(avg_score, on="id_student", how="left")
decoded_features = decoded_features.merge(num_assessments, on="id_student", how="left")
decoded_features = decoded_features.merge(total_clicks, on="id_student", how="left")
decoded_features = decoded_features.merge(active_days, on="id_student", how="left")
decoded_features = decoded_features.merge(avg_clicks_per_day, on="id_student", how="left")

# Add target column
decoded_features["target"] = decoded_features["final_result"].map(lambda x: 1 if x in ["Pass", "Distinction"] else 0)

# Save file as CSV after restoring original values
decoded_features.to_csv("decoded_student_data.csv", index=False)