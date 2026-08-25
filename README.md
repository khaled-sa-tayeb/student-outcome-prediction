<div align="center">

# 🎓 Student Outcome Prediction
### Machine Learning Classification Model

<p align="center">
  A machine learning classification project that predicts student academic outcomes (Pass or Fail) leveraging behavioral, demographic, and assessment data from the OULAD dataset.
</p>

<p align="center">
  <code><b>Python</b></code> &bull;
  <code><b>Scikit-Learn</b></code> &bull;
  <code><b>SMOTE</b></code> &bull;
  <code><b>Streamlit</b></code>
</p>

</div>

---

## 📊 Results & Performance

<div align="center">

| Model | Accuracy | Notes |
| :--- | :---: | :--- |
| **Logistic Regression** | 82% | Baseline linear model |
| **Random Forest (GridSearchCV)** | **85%** | Optimized via hyperparameter tuning |

</div>

---

## ⚙️ Project Pipeline

1. **Data Loading & Exploration:** Integrated 7 CSV files using Pandas to analyze shapes and missing values.
2. **Data Cleaning:** Handled missing `imd_band` values, dropped rows with missing assessment scores, encoded binary features, and filtered for binary classification (`Pass` / `Fail`).
3. **Feature Engineering:** 
   * **Total Clicks:** Sum of VLE interactions per student.
   * **Average Score:** Mean assessment score per student.
   * **Days Active:** Number of unique active days in the VLE.
   * **Withdrawal Flag:** Indicator for unregistration status.
4. **Model Training:** Applied SMOTE for class imbalance, split data using an 80/20 stratified approach, and optimized Random Forest.
5. **Deployment:** Saved the optimized model pipeline using `joblib` and developed a real-time web app.

---

## 📁 Project Structure

```text
student-outcome-prediction/
│
├── student.py                  # Main ML pipeline (cleaning, training, saving artifacts)
├── app.py                      # Interactive web application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── student_model.pkl           # Saved optimized model
├── model_features.pkl          # Feature columns reference list
│
└── data/                       # Directory for OULAD CSV files
    ├── studentInfo.csv
    ├── studentVle.csv
    ├── studentAssessment.csv
    ├── studentRegistration.csv
    ├── courses.csv
    ├── assessments.csv
    └── vle.csv
