import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# ========== App Page Configuration ==========
st.set_page_config(
    page_title="Student Success Prediction",
    page_icon="🎓",
    layout="wide"
)

# ========== CSS Styling ==========
st.markdown("""
    <style>
    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        font-size: 16px;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #e0e0e0;
        margin-top: 1rem;
    }
    .result-success {
        background-color: #e8f5e9;
        border-color: #81c784;
    }
    .result-fail {
        background-color: #ffebee;
        border-color: #e57373;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .risk-low {
        background-color: #c8e6c9;
        color: #1b5e20;
    }
    .risk-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
    }
    .compare-note {
        font-size: 0.9rem;
        color: #777;
        margin-top: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

# ========== Load Trained Model ==========
@st.cache_resource
def load_model():
    model = joblib.load("model/student_model.pkl")  # تم تحديث المسار ليكون داخل مجلد model
    return model

model = load_model()

# ========== Load Historical Student Data for Comparison ==========
@st.cache_data
def load_history():
    """
    Assumes decoded_student_data.csv exists in the same directory.
    Change the extension if you are using decoded_student_data.parquet or .pkl.
    """
    try:
        df = pd.read_csv("decoded_student_data.csv")
        return df
    except FileNotFoundError:
        return None

history_df = load_history()

# ========== Function to Calculate Percentile Rank Compared to Other Students ==========
def percentile_rank(series, value):
    # Percentage of students whose value is less than or equal to value
    return float((series <= value).mean() * 100.0)

# ========== Function to Generate Recommendations ==========
def generate_recommendations(
    avg_score,
    num_assessments,
    total_clicks,
    active_days,
    avg_clicks_per_day,
    withdrew,
    prediction,
    pass_proba
):
    recs = []

    if prediction == 0 or pass_proba < 70:
        recs.append("📌 Consider offering early academic support to help the student improve performance.")
    else:
        recs.append("✅ The student is performing well. Encourage them to maintain or increase their efforts.")

    if avg_score < 50:
        recs.append("🎯 The student may benefit from reviewing fundamental concepts and attending support sessions.")
    elif avg_score < 70:
        recs.append("📚 Encourage the student to focus more on quizzes, assignments, and regular study habits.")

    if num_assessments < 3:
        recs.append("📝 The student should aim to complete more assessments to improve their grade.")
    elif num_assessments < 6:
        recs.append("📝 Submitting more assessments could increase their chance of success.")

    if total_clicks < 2000:
        recs.append("💻 Low platform activity detected. Encourage more engagement with course materials.")
    if active_days < 40:
        recs.append("📆 The student should increase the number of active study days to build consistency.")
    if avg_clicks_per_day < 15:
        recs.append("⌛ Increasing daily interaction with course content may help improve outcomes.")

    if withdrew == 1:
        recs.append("⚠️ The student has withdrawn previously. Consider reviewing reasons and offering support.")

    if not recs:
        recs.append("✅ No major concerns detected. The student seems on track.")

    return recs

# ========== Page Titles ==========
st.markdown('<div class="main-title">🎓 Student Success Prediction App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    "Enter the student's information in the sidebar to predict their likelihood of success in the course, "
    "along with detailed analytics, comparison with other students, and personalized recommendations."
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ========== Sidebar Data Input ==========
with st.sidebar:
    st.header("📋 Student Information")

    st.subheader("👤 Personal Info")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age_band = st.selectbox("Age Band", ["0-35", "35-55", "55<"])
    highest_education = st.selectbox("Highest Education", [
        "Lower Than A Level",
        "A Level or Equivalent",
        "HE Qualification",
        "Post Graduate Qualification"
    ])
    region = st.selectbox("Region", [
        "East Anglian Region", "East Midlands Region", "Ireland", "London Region",
        "North Region", "North Western Region", "Scotland", "South East Region",
        "South Region", "South West Region", "Wales", "West Midlands Region", "Yorkshire Region"
    ])
    disability = st.selectbox("Disability", ["No", "Yes"])
    imd_band = st.selectbox("Deprivation Index Band", [
        "0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
        "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"
    ])

    st.subheader("📚 Academic & Engagement Data")
    avg_score = st.slider("Average Score", 0, 100, 70)
    num_assessments = st.slider("Assessments Submitted", 0, 30, 5)
    total_clicks = st.slider("Total VLE Clicks", 0, 10000, 1000, step=100)
    active_days = st.slider("Active Days on VLE", 0, 250, 50)
    avg_clicks_per_day = st.slider("Avg Clicks Per Day", 0.0, 500.0, 20.0)
    withdrew = st.selectbox("Withdrawn?", [0, 1])

    predict_button = st.button("🔍 Predict Outcome", use_container_width=True)

# ========== Prepare Data for the Model ==========
input_data = {
    'avg_score': avg_score,
    'num_assessments': num_assessments,
    'total_clicks': total_clicks,
    'active_days': active_days,
    'avg_clicks_per_day': avg_clicks_per_day,
    'withdrew': withdrew,
    'disability': 1 if disability == "Yes" else 0,
    'gender_Male': 1 if gender == "Male" else 0,
    'region_' + region: 1,
    'highest_education_' + highest_education: 1,
    'age_band_' + age_band: 1,
    'imd_band_' + imd_band: 1
}

all_features = model.feature_names_in_
row = pd.DataFrame([input_data])

for col in all_features:
    if col not in row.columns:
        row[col] = 0

row = row[all_features]

# ========== Page Layout ==========
col_main, col_side = st.columns([2, 1])

with col_main:
    st.subheader("🔎 Prediction Result")

    if predict_button:
        prediction = model.predict(row)[0]
        proba = model.predict_proba(row)[0]

        pass_proba = proba[1] * 100
        fail_proba = proba[0] * 100

        if prediction == 1:
            result_text = "✅ The student is likely to pass the course."
            card_class = "result-success"
            confidence = pass_proba

            if pass_proba >= 80:
                risk_label = "Low Risk"
                risk_class = "risk-low"
            elif pass_proba >= 60:
                risk_label = "Medium Risk"
                risk_class = "risk-medium"
            else:
                risk_label = "High Risk"
                risk_class = "risk-high"
        else:
            result_text = "❌ The student is at risk of failing the course."
            card_class = "result-fail"
            confidence = fail_proba

            if fail_proba >= 80:
                risk_label = "High Risk"
                risk_class = "risk-high"
            elif fail_proba >= 60:
                risk_label = "Medium Risk"
                risk_class = "risk-medium"
            else:
                risk_label = "Low Risk"
                risk_class = "risk-low"

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <h3>Predicted Outcome</h3>
                <p>{result_text}</p>
                <div class="risk-badge {risk_class}">{risk_label}</div>
                <p style="margin-top: 0.8rem; color: #555;">
                    Confidence: <b>{confidence:.2f}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ========== Statistics + Comparison with Other Students ==========
        st.markdown("### 📊 Engagement & Performance Statistics (Compared to Other Students)")

        engagement_score = min(
            100,
            (total_clicks / 5000) * 40 +
            (active_days / 180) * 30 +
            (avg_clicks_per_day / 100) * 30
        )
        assessment_score = min(100, (num_assessments / 10) * 100)
        performance_score = avg_score

        # Is comparison data available?
        if history_df is not None:
            compare_cols = ["avg_score", "num_assessments", "total_clicks", "active_days", "avg_clicks_per_day"]
            missing_cols = [c for c in compare_cols if c not in history_df.columns]
            history_available = len(missing_cols) == 0
        else:
            missing_cols = []
            history_available = False

        col1, col2, col3 = st.columns(3)

        if history_available:
            avg_score_pct = percentile_rank(history_df["avg_score"], avg_score)
            num_assessments_pct = percentile_rank(history_df["num_assessments"], num_assessments)
            total_clicks_pct = percentile_rank(history_df["total_clicks"], total_clicks)
            active_days_pct = percentile_rank(history_df["active_days"], active_days)
            avg_clicks_day_pct = percentile_rank(history_df["avg_clicks_per_day"], avg_clicks_per_day)

            hist_engagement = (
                (history_df["total_clicks"] / 5000) * 40 +
                (history_df["active_days"] / 180) * 30 +
                (history_df["avg_clicks_per_day"] / 100) * 30
            ).clip(0, 100)
            engagement_pct = percentile_rank(hist_engagement, engagement_score)

            with col1:
                st.metric("Performance Score", f"{performance_score:.0f} / 100")
                st.markdown(
                    f"<div class='compare-note'>Higher than ~{avg_score_pct:.0f}% of students.</div>",
                    unsafe_allow_html=True
                )
            with col2:
                st.metric("Engagement Score", f"{engagement_score:.0f} / 100")
                st.markdown(
                    f"<div class='compare-note'>Higher than ~{engagement_pct:.0f}% of students.</div>",
                    unsafe_allow_html=True
                )
            with col3:
                st.metric("Assessments Score", f"{assessment_score:.0f} / 100")
                st.markdown(
                    f"<div class='compare-note'>More assessments than ~{num_assessments_pct:.0f}% of students.</div>",
                    unsafe_allow_html=True
                )

            st.markdown("### 📈 Probability Distribution")
            stats_df = pd.DataFrame({
                "Outcome": ["Success", "Fail"],
                "Probability": [pass_proba, fail_proba]
            }).set_index("Outcome")
            st.bar_chart(stats_df)

            st.markdown("### 📊 Additional Comparisons")
            col_extra1, col_extra2 = st.columns(2)
            with col_extra1:
                st.write(f"- Total VLE Clicks: higher than ~{total_clicks_pct:.0f}% of students.")
                st.write(f"- Active Days on VLE: higher than ~{active_days_pct:.0f}% of students.")
            with col_extra2:
                st.write(f"- Avg Clicks per Day: higher than ~{avg_clicks_day_pct:.0f}% of students.")
        else:
            with col1:
                st.metric("Performance Score", f"{performance_score:.0f} / 100")
            with col2:
                st.metric("Engagement Score", f"{engagement_score:.0f} / 100")
            with col3:
                st.metric("Assessments Score", f"{assessment_score:.0f} / 100")

            if history_df is None:
                st.info("Comparison data not found. Make sure 'decoded_student_data.csv' exists.")
            elif missing_cols:
                st.warning(f"Missing required columns in 'decoded_student_data.csv': {missing_cols}")

        # ========== Recommendations ==========
        st.markdown("### 💡 Personalized Recommendations")
        recommendations = generate_recommendations(
            avg_score=avg_score,
            num_assessments=num_assessments,
            total_clicks=total_clicks,
            active_days=active_days,
            avg_clicks_per_day=avg_clicks_per_day,
            withdrew=withdrew,
            prediction=prediction,
            pass_proba=pass_proba
        )

        for rec in recommendations:
            st.markdown(f"- {rec}")

with col_side:
    st.subheader("📈 Quick Metrics")

    if predict_button:
        st.metric("Success Probability", f"{pass_proba:.1f} %")
        st.metric("Fail Probability", f"{fail_proba:.1f} %")
        st.metric("Average Score", f"{avg_score} / 100")
        st.metric("Total Clicks", f"{total_clicks}")
    else:
        st.info("Enter data and click 'Predict' to see results.")