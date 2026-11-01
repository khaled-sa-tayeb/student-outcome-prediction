import pandas as pd

# Shared by student.py (training) and app.py (serving) so both encode inputs identically

KEYS = ["id_student", "code_module", "code_presentation"]

CATEGORICAL = ["code_module", "gender", "region", "highest_education", "age_band", "imd_band"]

NUMERIC = [
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "avg_score",
    "num_assessments",
    "total_clicks",
    "active_days",
    "clicks_per_active_day",
    "clicks_last_14_days",
    "days_since_last_activity",
    "missed_assessments",
    "avg_days_late",
]


def encode(df, feature_columns=None):
    X = df[NUMERIC + CATEGORICAL].copy()
    X["disability"] = X["disability"].map({"Y": 1, "N": 0})
    X = pd.get_dummies(X, columns=CATEGORICAL, dtype=int)

    # Align to the training columns: unseen categories are dropped, missing ones become 0
    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0)

    return X.fillna(0)
