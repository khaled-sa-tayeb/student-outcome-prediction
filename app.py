import json
from html import escape
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from features import encode

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "Model"
GITHUB_URL = "https://github.com/khaled-sa-tayeb/student-outcome-prediction"

st.set_page_config(
    page_title="Student Success Predictor",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================== Styling ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --surface: #FFFFFF;
    --border: #E4E7EF;
    --text: #1B1F2A;
    --muted: #667085;
    --primary: #4F46E5;
    --primary-soft: #EEF0FF;
    --good: #12B76A;
    --good-soft: #E7F8EF;
    --warn: #F79009;
    --warn-soft: #FEF4E6;
    --bad: #F04438;
    --bad-soft: #FDECEB;
}

html, body, [class*="css"], .stMarkdown, button, input, label {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* App chrome: hide Streamlit's header, menu, footer and sidebar */
header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stSidebar"],
[data-testid="collapsedControl"], .stDeployButton { display: none !important; }

.block-container { max-width: 1180px; padding: 1.2rem 1.2rem 3rem; }

/* Top bar */
.topbar {
    display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
    background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
    padding: 0.9rem 1.2rem; margin-bottom: 1.2rem; box-shadow: 0 1px 2px rgba(16,24,40,.04);
}
.brand { display: flex; align-items: center; gap: 0.8rem; }
.brand-icon {
    width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center;
    background: linear-gradient(135deg, #4F46E5, #7C3AED); color: white;
}
.brand-title { font-size: 1.15rem; font-weight: 800; color: var(--text); line-height: 1.2; }
.brand-sub { font-size: 0.85rem; color: var(--muted); }
.top-link {
    font-size: 0.85rem; font-weight: 600; color: var(--text) !important; text-decoration: none !important;
    border: 1px solid var(--border); border-radius: 10px; padding: 0.45rem 0.8rem; background: var(--surface);
}
.top-link:hover { border-color: var(--primary); color: var(--primary) !important; }

/* Section labels */
.eyebrow { font-size: 0.75rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 0 0 .4rem; }
.hint { font-size: 0.85rem; color: var(--muted); margin-top: .2rem; }

/* Cards (containers created with key="card_N") */
[class*="st-key-card_"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
    padding: 1rem 1.2rem; box-shadow: 0 1px 2px rgba(16,24,40,.04);
}

/* Result card */
.result { display: flex; gap: 1.4rem; align-items: center; flex-wrap: wrap; }
.ring { position: relative; width: 132px; height: 132px; flex: 0 0 auto; }
.ring svg { transform: rotate(-90deg); }
.ring-label { position: absolute; inset: 0; display: grid; place-items: center; text-align: center; }
.ring-value { font-size: 1.9rem; font-weight: 800; color: var(--text); line-height: 1; }
.ring-caption { font-size: 0.72rem; color: var(--muted); margin-top: .25rem; }
.verdict-chip { display: inline-block; font-size: 0.8rem; font-weight: 700; padding: .3rem .7rem; border-radius: 999px; }
.verdict-title { font-size: 1.35rem; font-weight: 800; color: var(--text); margin: .5rem 0 .3rem; }
.verdict { flex: 1 1 220px; min-width: 0; }
.verdict-text { font-size: 0.95rem; color: var(--muted); line-height: 1.5; }

/* Factor and comparison rows */
.row { display: flex; align-items: center; justify-content: space-between; gap: .8rem; padding: .55rem 0; border-bottom: 1px solid #F0F2F7; }
.row:last-child { border-bottom: none; }
.row > div:first-child { flex: 1 1 auto; min-width: 0; }
.row-name { font-size: 0.9rem; color: var(--text); font-weight: 500; }
.row-value { font-size: 0.8rem; color: var(--muted); }
.delta { font-size: 0.85rem; font-weight: 700; white-space: nowrap; }
.bar { flex: 0 0 42%; height: 8px; background: #EEF0F6; border-radius: 99px; overflow: hidden; }
.bar > div { height: 100%; border-radius: 99px; }
.pct { font-size: 0.8rem; font-weight: 700; width: 3.2rem; text-align: right; }

/* Recommendation cards */
.rec { display: flex; gap: .8rem; padding: .8rem; border-radius: 12px; background: #F8F9FC; margin-bottom: .55rem; }
.rec-icon { flex: 0 0 34px; width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center; }

/* Example buttons: colored icons */
.st-key-ex_on_track [data-testid="stIconMaterial"] { color: var(--good); }
.st-key-ex_borderline [data-testid="stIconMaterial"] { color: var(--warn); }
.st-key-ex_at_risk [data-testid="stIconMaterial"] { color: var(--bad); }
.rec-title { font-size: 0.92rem; font-weight: 700; color: var(--text); }
.rec-text { font-size: 0.85rem; color: var(--muted); line-height: 1.45; }

/* Stat tiles */
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .8rem; margin-bottom: .6rem; }
.tile { background: #F8F9FC; border-radius: 12px; padding: .8rem 1rem; }
.tile-value { font-size: 1.35rem; font-weight: 800; color: var(--text); }
.tile-label { font-size: 0.8rem; color: var(--muted); }

/* Phones: live result pinned to the bottom while scrolling through the inputs */
.mobile-bar { display: none; }
@media (max-width: 640px) {
    .block-container { padding-bottom: 6rem; }
    .mobile-bar {
        display: flex; align-items: center; justify-content: space-between; gap: .8rem;
        position: fixed; left: 12px; right: 12px; bottom: 12px; z-index: 1000;
        background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
        padding: .7rem 1rem; box-shadow: 0 8px 24px rgba(16,24,40,.14);
    }
    .mobile-bar-value { font-size: 1.4rem; font-weight: 800; color: var(--text); }
    .mobile-bar-label { font-size: .75rem; color: var(--muted); }
    .bar { flex-basis: 34%; }
}

.footer { text-align: center; font-size: 0.82rem; color: var(--muted); margin-top: 1.5rem; }
.footer a { color: var(--primary) !important; }

/* Widgets */
.stTabs [data-baseweb="tab-list"] { gap: .3rem; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
div[data-testid="stSegmentedControl"] button { font-weight: 600; }
.stButton > button { border-radius: 10px; font-weight: 600; white-space: nowrap; padding-left: .6rem; padding-right: .6rem; }
.stButton > button p { white-space: nowrap; }
</style>
""", unsafe_allow_html=True)


# ============================== Data & Models ==============================
@st.cache_data(show_spinner=False)
def load_summary():
    with open(MODEL_DIR / "summary.json") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_model(day):
    folder = MODEL_DIR / f"day_{day}"
    model = joblib.load(folder / "student_model.pkl")
    feature_columns = joblib.load(folder / "feature_columns.pkl")
    with open(folder / "metrics.json") as f:
        metrics = json.load(f)
    return model, feature_columns, metrics


@st.cache_data(show_spinner=False)
def load_history():
    return pd.read_csv(BASE_DIR / "decoded_student_data.csv")


summary = load_summary()
checkpoint_days = [m["cutoff_day"] for m in summary]
history = load_history()

# ============================== Inputs Definition ==============================
# (label, help, higher_is_better)
ACTIVITY = {
    "avg_score": ("Average score", "Mean score of assessments submitted so far (0–100).", True),
    "num_assessments": ("Assessments submitted", "How many assessments the student has handed in so far.", True),
    "missed_assessments": ("Missed deadlines", "Assessments already due that the student has not submitted.", False),
    "avg_days_late": ("Avg. days late", "Average days between deadline and submission. Negative means early.", False),
    "total_clicks": ("Platform clicks", "Total clicks on the learning platform since the course started.", True),
    "active_days": ("Active days", "Number of days the student used the learning platform.", True),
    "clicks_last_14_days": ("Clicks, last 14 days", "Recent activity: platform clicks in the last two weeks.", True),
    "days_since_last_activity": ("Days since last login", "Days since the student last used the platform.", False),
}

PROFILE_DEFAULTS = {
    "code_module": "AAA",
    "num_of_prev_attempts": 0,
    "studied_credits": 60,
    "gender": "Female",
    "age_band": "0-35",
    "highest_education": "A Level or Equivalent",
    "region": "London Region",
    "imd_band": "50-60%",
    "disability": "No",
}

# Example students, defined by where they sit among classmates at the same course day
EXAMPLES = {
    "typical": {"q": 0.50, "avg_score": None, "missed_assessments": 0, "avg_days_late": 0, "days_since_last_activity": None},
    "on_track": {"q": 0.80, "avg_score": 88, "missed_assessments": 0, "avg_days_late": -2, "days_since_last_activity": 1},
    "borderline": {"q": 0.35, "avg_score": 62, "missed_assessments": 1, "avg_days_late": 2, "days_since_last_activity": 6},
    "at_risk": {"q": 0.12, "avg_score": 42, "missed_assessments": 2, "avg_days_late": 5, "days_since_last_activity": 21},
}


def peers(day):
    return history[history["checkpoint_day"] == day]


def slider_max(day, col, step):
    top = peers(day)[col].quantile(0.99)
    return max(step, int(-(-top // step) * step))


def example_values(day, name):
    ex = EXAMPLES[name]
    p = peers(day)
    window = day + 30
    values = {
        "num_assessments": int(round(p["num_assessments"].quantile(ex["q"]))),
        "total_clicks": int(round(p["total_clicks"].quantile(ex["q"]) / 50) * 50),
        "active_days": int(round(p["active_days"].quantile(ex["q"]))),
        "clicks_last_14_days": int(round(p["clicks_last_14_days"].quantile(ex["q"]) / 10) * 10),
        "missed_assessments": ex["missed_assessments"],
        "avg_days_late": ex["avg_days_late"],
    }
    values["avg_score"] = ex["avg_score"] if ex["avg_score"] is not None else int(round(p["avg_score"].median()))
    values["days_since_last_activity"] = min(
        window, ex["days_since_last_activity"] if ex["days_since_last_activity"] is not None
        else int(round(p["days_since_last_activity"].median()))
    )
    values["total_clicks"] = min(values["total_clicks"], slider_max(day, "total_clicks", 50))
    values["clicks_last_14_days"] = min(values["clicks_last_14_days"], slider_max(day, "clicks_last_14_days", 10))
    values["num_assessments"] = min(values["num_assessments"], slider_max(day, "num_assessments", 1))
    values["active_days"] = min(values["active_days"], window)
    return values


def apply_example(name):
    st.session_state.update(example_values(st.session_state.day, name))


def on_day_change():
    # Activity totals are cumulative, so reset them to typical values for the new day
    if st.session_state.day is None:
        st.session_state.day = st.session_state.last_day
    st.session_state.last_day = st.session_state.day
    apply_example("typical")


if "day" not in st.session_state:
    st.session_state.day = 60 if 60 in checkpoint_days else checkpoint_days[0]
    st.session_state.last_day = st.session_state.day
    st.session_state.update(PROFILE_DEFAULTS)
    apply_example("typical")


# ============================== Prediction Helpers ==============================
def make_consistent(values, day):
    v = dict(values)
    if v["total_clicks"] == 0 or v["active_days"] == 0:
        v["clicks_last_14_days"] = 0
        v["days_since_last_activity"] = day + 30
    v["clicks_last_14_days"] = min(v["clicks_last_14_days"], v["total_clicks"])
    return v


def predict(values, profile, day):
    model, feature_columns, _ = load_model(day)
    v = make_consistent(values, day)
    v["clicks_per_active_day"] = v["total_clicks"] / v["active_days"] if v["active_days"] else 0.0
    row = pd.DataFrame([{
        **v,
        "code_module": profile["code_module"],
        "gender": "M" if profile["gender"] == "Male" else "F",
        "region": profile["region"],
        "highest_education": profile["highest_education"],
        "age_band": profile["age_band"],
        "imd_band": profile["imd_band"],
        "num_of_prev_attempts": profile["num_of_prev_attempts"],
        "studied_credits": profile["studied_credits"],
        "disability": "Y" if profile["disability"] == "Yes" else "N",
    }])
    return float(model.predict_proba(encode(row, feature_columns))[0, 1])


def standing(day, col, value, higher_is_better):
    """Share of classmates (same course day) this student is doing at least as well as."""
    s = peers(day)[col]
    share = (s <= value).mean() if higher_is_better else (s >= value).mean()
    return float(share * 100)


# Line icons (Lucide, MIT licensed), drawn with the current text color
ICONS = {
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "message": '<path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "book": '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "monitor": '<rect width="20" height="14" x="2" y="3" rx="2"/><line x1="8" x2="16" y1="21" y2="21"/><line x1="12" x2="12" y1="17" y2="21"/>',
    "trend_down": '<polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
}


def icon_svg(name, size=18):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


def tone(pct):
    if pct >= 40:
        return "var(--good)"
    if pct >= 20:
        return "var(--warn)"
    return "var(--bad)"


# ============================== Top Bar ==============================
st.markdown(f"""
<div class="topbar">
<div class="brand">
<div class="brand-icon"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10 12 5 2 10l10 5 10-5z"/><path d="M6 12v5c3 2 9 2 12 0v-5"/></svg></div>
<div><div class="brand-title">Student Success Predictor</div>
<div class="brand-sub">Early-warning model for online courses · trained on 26K+ real enrollments</div></div>
</div>
<a class="top-link" href="{GITHUB_URL}" target="_blank">View on GitHub ↗</a>
</div>
""", unsafe_allow_html=True)

# ============================== Course Day + Examples ==============================
with st.container(key="card_1"):
    st.markdown('<p class="eyebrow">Where is the course now?</p>', unsafe_allow_html=True)
    st.segmented_control(
        "Course day", checkpoint_days, key="day", on_change=on_day_change,
        format_func=lambda d: f"Day {d}", label_visibility="collapsed",
    )
    st.markdown('<p class="eyebrow" style="margin-top:.6rem">Try an example student</p>', unsafe_allow_html=True)
    b1, b2, b3, _ = st.columns([1, 1, 1, 1.5])
    b1.button("On track", icon=":material/check_circle:", key="ex_on_track",
              on_click=apply_example, args=("on_track",), use_container_width=True)
    b2.button("Borderline", icon=":material/error:", key="ex_borderline",
              on_click=apply_example, args=("borderline",), use_container_width=True)
    b3.button("At risk", icon=":material/warning:", key="ex_at_risk",
              on_click=apply_example, args=("at_risk",), use_container_width=True)

day = st.session_state.day
window = day + 30
_, _, metrics = load_model(day)

st.markdown(
    f'<p class="hint">Using the <b>day {day}</b> model: it only sees what is known by day {day} of the course '
    f'and was right for <b>{metrics["accuracy"] * 100:.0f}%</b> of students it had never seen. '
    "Results update as you change the inputs.</p>",
    unsafe_allow_html=True,
)

left, right = st.columns([5, 7], gap="large")

# ============================== Inputs ==============================
with left:
    with st.container(key="card_2"):
        st.markdown('<p class="eyebrow">Student details</p>', unsafe_allow_html=True)
        tab_activity, tab_course, tab_profile = st.tabs([
            ":material/bar_chart: Activity", ":material/school: Course", ":material/person: Profile",
        ])

        with tab_activity:
            st.caption(f"Totals from the start of the course up to day {day}.")
            st.slider(ACTIVITY["avg_score"][0], 0, 100, key="avg_score", help=ACTIVITY["avg_score"][1])
            a, b = st.columns(2)
            a.slider(ACTIVITY["num_assessments"][0], 0, slider_max(day, "num_assessments", 1),
                     key="num_assessments", help=ACTIVITY["num_assessments"][1])
            b.slider(ACTIVITY["missed_assessments"][0], 0, 5, key="missed_assessments",
                     help=ACTIVITY["missed_assessments"][1])
            st.slider(ACTIVITY["avg_days_late"][0], -14, 14, key="avg_days_late", help=ACTIVITY["avg_days_late"][1])
            st.slider(ACTIVITY["total_clicks"][0], 0, slider_max(day, "total_clicks", 50), step=50,
                      key="total_clicks", help=ACTIVITY["total_clicks"][1])
            st.slider(ACTIVITY["active_days"][0], 0, window, key="active_days", help=ACTIVITY["active_days"][1])
            st.slider(ACTIVITY["clicks_last_14_days"][0], 0, slider_max(day, "clicks_last_14_days", 10), step=10,
                      key="clicks_last_14_days", help=ACTIVITY["clicks_last_14_days"][1])
            st.slider(ACTIVITY["days_since_last_activity"][0], 0, window, key="days_since_last_activity",
                      help=ACTIVITY["days_since_last_activity"][1])

        with tab_course:
            st.selectbox("Module", ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"], key="code_module",
                         help="Course module code in the Open University dataset.")
            st.slider("Previous attempts at this module", 0, 6, key="num_of_prev_attempts")
            st.slider("Credits studied this term", 30, 300, step=15, key="studied_credits",
                      help="A heavier study load leaves less time for each course.")

        with tab_profile:
            a, b = st.columns(2)
            a.selectbox("Gender", ["Female", "Male"], key="gender")
            b.selectbox("Age band", ["0-35", "35-55", "55<="], key="age_band")
            st.selectbox("Highest education", [
                "No Formal quals", "Lower Than A Level", "A Level or Equivalent",
                "HE Qualification", "Post Graduate Qualification",
            ], key="highest_education")
            st.selectbox("Region", [
                "East Anglian Region", "East Midlands Region", "Ireland", "London Region",
                "North Region", "North Western Region", "Scotland", "South East Region",
                "South Region", "South West Region", "Wales", "West Midlands Region", "Yorkshire Region",
            ], key="region")
            a, b = st.columns(2)
            a.selectbox("Deprivation index", [
                "0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
            ], key="imd_band", help="UK Index of Multiple Deprivation band of the student's area (0-10% = most deprived).")
            b.selectbox("Disability", ["No", "Yes"], key="disability")

# ============================== Prediction ==============================
values = {k: st.session_state[k] for k in ACTIVITY}
profile = {k: st.session_state[k] for k in PROFILE_DEFAULTS}
threshold = metrics.get("threshold", 0.5)
pass_proba = predict(values, profile, day)

if pass_proba < threshold:
    verdict, chip_bg, color = "At risk", "var(--bad-soft)", "var(--bad)"
    title = "Likely to fail or withdraw"
    text = "This student matches the pattern of students who did not pass. Reaching out now gives the most time to help."
elif pass_proba < 0.75:
    verdict, chip_bg, color = "Monitor", "var(--warn-soft)", "var(--warn)"
    title = "Likely to pass, but not safely"
    text = "The model leans towards a pass, with real uncertainty. Keep an eye on deadlines and activity."
else:
    verdict, chip_bg, color = "On track", "var(--good-soft)", "var(--good)"
    title = "Likely to pass"
    text = "Activity and results look like students who passed the course."

st.markdown(f"""
<div class="mobile-bar">
<div><div class="mobile-bar-value">{pass_proba * 100:.0f}%</div><div class="mobile-bar-label">chance to pass · day {day} model</div></div>
<span class="verdict-chip" style="background:{chip_bg};color:{color}">{verdict}</span>
</div>
""", unsafe_allow_html=True)

# Which inputs move the prediction most, compared with a typical classmate at this day
typical = example_values(day, "typical")
factors = []
for key in ACTIVITY:
    if values[key] == typical[key]:
        continue
    alt_values = dict(values, **{key: typical[key]})
    delta = pass_proba - predict(alt_values, profile, day)
    if abs(delta) >= 0.01:
        factors.append((key, delta))
factors.sort(key=lambda f: abs(f[1]), reverse=True)

# ============================== Results ==============================
with right:
    with st.container(key="card_3"):
        circumference = 2 * 3.14159 * 54
        filled = circumference * pass_proba
        st.markdown(f"""
<div class="result">
<div class="ring">
<svg width="132" height="132" viewBox="0 0 132 132">
<circle cx="66" cy="66" r="54" fill="none" stroke="#EEF0F6" stroke-width="12"/>
<circle cx="66" cy="66" r="54" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round"
stroke-dasharray="{filled:.1f} {circumference:.1f}"/>
</svg>
<div class="ring-label"><div><div class="ring-value">{pass_proba * 100:.0f}%</div><div class="ring-caption">chance to pass</div></div></div>
</div>
<div class="verdict">
<span class="verdict-chip" style="background:{chip_bg};color:{color}">{verdict}</span>
<div class="verdict-title">{title}</div>
<div class="verdict-text">{text}</div>
</div>
</div>
""", unsafe_allow_html=True)

    with st.container(key="card_4"):
        st.markdown('<p class="eyebrow">What is driving this result</p>', unsafe_allow_html=True)
        if factors:
            rows = []
            for key, delta in factors[:4]:
                label = ACTIVITY[key][0]
                arrow, col = ("▲", "var(--good)") if delta > 0 else ("▼", "var(--bad)")
                rows.append(
                    f'<div class="row"><div><div class="row-name">{escape(label)}: {values[key]}</div>'
                    f'<div class="row-value">typical classmate: {typical[key]}</div></div>'
                    f'<div class="delta" style="color:{col}">{arrow} {abs(delta) * 100:.0f} pts</div></div>'
                )
            st.markdown("".join(rows), unsafe_allow_html=True)
            st.caption("Change in pass chance if this one input were typical for the class, everything else kept the same.")
        else:
            st.markdown('<p class="hint">This student looks like a typical classmate at this point of the course.</p>',
                        unsafe_allow_html=True)

    with st.container(key="card_5"):
        st.markdown(f'<p class="eyebrow">Compared with classmates at day {day}</p>', unsafe_allow_html=True)
        rows = []
        for key, (label, _, higher_is_better) in ACTIVITY.items():
            pct = standing(day, key, values[key], higher_is_better)
            rows.append(
                f'<div class="row"><div><div class="row-name">{escape(label)}</div>'
                f'<div class="row-value">{values[key]}</div></div>'
                f'<div class="bar"><div style="width:{pct:.0f}%;background:{tone(pct)}"></div></div>'
                f'<div class="pct" style="color:{tone(pct)}">{pct:.0f}%</div></div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)
        st.caption("Share of classmates this student is doing at least as well as.")

    # ---------- Recommendations ----------
    recs = []
    if values["missed_assessments"] > 0:
        recs.append(("alert", "bad", "Follow up on missed work",
                     f"{values['missed_assessments']} deadline(s) passed without a submission. Agree on a catch-up plan."))
    if values["days_since_last_activity"] >= 14:
        recs.append(("message", "bad", "Check in personally",
                     f"No platform activity for {values['days_since_last_activity']} days, a common sign before withdrawing."))
    if values["avg_score"] < 50:
        recs.append(("target", "warn", "Strengthen the basics",
                     "Scores are below 50. Suggest support sessions and reviewing core concepts."))
    elif values["avg_score"] < 70:
        recs.append(("book", "warn", "Build steadier study habits",
                     "Scores are moderate. Regular practice on quizzes and assignments can lift them."))
    if values["avg_days_late"] > 0:
        recs.append(("clock", "warn", "Plan ahead of deadlines",
                     "Work is usually handed in late. A simple weekly plan can help."))
    if standing(day, "total_clicks", values["total_clicks"], True) < 25:
        recs.append(("monitor", "primary", "Encourage more engagement",
                     "Platform activity is in the bottom quarter of the class."))
    if standing(day, "clicks_last_14_days", values["clicks_last_14_days"], True) < 25:
        recs.append(("trend_down", "primary", "Activity is dropping",
                     "Very little activity in the last two weeks compared with classmates."))
    if not recs:
        recs.append(("check", "good", "Keep it up", "No warning signs right now. Encourage the student to keep the same pace."))

    with st.container(key="card_6"):
        st.markdown('<p class="eyebrow">Recommended actions</p>', unsafe_allow_html=True)
        st.markdown("".join(
            f'<div class="rec"><div class="rec-icon" style="background:var(--{tone_name}-soft);color:var(--{tone_name})">'
            f'{icon_svg(icon)}</div><div><div class="rec-title">{escape(t)}</div>'
            f'<div class="rec-text">{escape(body)}</div></div></div>'
            for icon, tone_name, t, body in recs
        ), unsafe_allow_html=True)

# ============================== How It Works ==============================
st.markdown('<p class="eyebrow" style="margin-top:1.2rem">How the model works</p>', unsafe_allow_html=True)
with st.container(key="card_7"):
    last = summary[-1]
    st.markdown(f"""
<div class="tiles">
<div class="tile"><div class="tile-value">10.6M</div><div class="tile-label">platform activity records</div></div>
<div class="tile"><div class="tile-value">26K+</div><div class="tile-label">course enrollments</div></div>
<div class="tile"><div class="tile-value">{len(summary)}</div><div class="tile-label">models, one per course checkpoint</div></div>
<div class="tile"><div class="tile-value">{last['accuracy'] * 100:.0f}%</div><div class="tile-label">accuracy at day {last['cutoff_day']} on unseen students</div></div>
</div>
""", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        chart_df = pd.DataFrame([
            {"Course day": m["cutoff_day"], "Metric": name, "Value": m[key] * 100}
            for m in summary
            for key, name in [("accuracy", "Accuracy"), ("at_risk_recall", "At-risk students caught")]
        ])
        chart = (
            alt.Chart(chart_df)
            .mark_line(point=alt.OverlayMarkDef(size=60, filled=True), strokeWidth=3)
            .encode(
                x=alt.X("Course day:Q", scale=alt.Scale(domain=[checkpoint_days[0], checkpoint_days[-1]]),
                        axis=alt.Axis(values=checkpoint_days, title="Course day", grid=False)),
                y=alt.Y("Value:Q", scale=alt.Scale(domain=[50, 100]), axis=alt.Axis(title="%", tickCount=6)),
                color=alt.Color("Metric:N", scale=alt.Scale(range=["#4F46E5", "#F79009"]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["Metric", "Course day", alt.Tooltip("Value:Q", format=".1f")],
            )
            .properties(height=260)
            .configure(background="#FFFFFF")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor="#667085", titleColor="#667085", domainColor="#E4E7EF", gridColor="#F0F2F7")
            .configure_legend(labelColor="#1B1F2A")
        )
        st.altair_chart(chart, use_container_width=True, theme=None)
    with c2:
        st.markdown("""
**Earlier warning or higher accuracy.** Each checkpoint has its own model that only sees data
available by that day. Later checkpoints are more accurate; earlier ones leave more time to help.

**What it looks at:** assessment scores, missed and late submissions, platform activity and how
recent it is, plus the course and the student's background.

**How it was tested:** on students held out from training, the same students at every checkpoint.

**Data:** the public Open University Learning Analytics Dataset (OULAD). This is a demo for
learning purposes, not a tool for real decisions about students.
""")

st.markdown(
    f'<div class="footer">Built by Khaled Tayeb · <a href="{GITHUB_URL}" target="_blank">Source code</a></div>',
    unsafe_allow_html=True,
)
