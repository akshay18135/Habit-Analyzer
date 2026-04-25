from pathlib import Path
import time

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "habit_data.csv"
FEATURE_COLUMNS = ["Day", "Time", "Activity", "Mood", "Duration"]
CATEGORY_COLUMNS = ["Day", "Time", "Activity", "Mood", "Category"]
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TIME_ORDER = ["Morning", "Afternoon", "Evening", "Night"]
PLOTLY_TEMPLATE = "plotly_dark"


st.set_page_config(page_title="Habit Analyzer Pro", layout="wide")

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "last_prediction_signature" not in st.session_state:
    st.session_state.last_prediction_signature = None


def apply_dashboard_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #0e1117;
                --surface: #1c1f26;
                --surface-2: #151922;
                --glass: rgba(28, 31, 38, 0.78);
                --border: rgba(148, 163, 184, 0.18);
                --text: #f8fafc;
                --muted: #a7b0c0;
                --blue: #38bdf8;
                --green: #22c55e;
                --orange: #f59e0b;
                --red: #ef4444;
                --purple: #a78bfa;
            }

            .stApp {
                background:
                    radial-gradient(circle at 8% 4%, rgba(56, 189, 248, 0.16), transparent 28rem),
                    radial-gradient(circle at 92% 5%, rgba(34, 197, 94, 0.12), transparent 30rem),
                    radial-gradient(circle at 42% 48%, rgba(167, 139, 250, 0.07), transparent 34rem),
                    var(--bg);
                color: var(--text);
                font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }

            .block-container {
                max-width: 1380px;
                padding: 2rem 2.2rem 3.2rem;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(28, 31, 38, 0.97), rgba(15, 23, 42, 0.97));
                border-right: 1px solid var(--border);
                box-shadow: 18px 0 42px rgba(0, 0, 0, 0.30);
            }

            [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
                gap: 1rem;
            }

            h1, h2, h3, h4, label {
                color: var(--text) !important;
            }

            p, [data-testid="stMarkdownContainer"] p {
                color: var(--muted);
            }

            .hero {
                background:
                    linear-gradient(135deg, rgba(14, 165, 233, 0.95), rgba(34, 197, 94, 0.72)),
                    linear-gradient(45deg, rgba(30, 41, 59, 0.94), rgba(15, 23, 42, 0.94));
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 24px;
                box-shadow: 0 28px 80px rgba(0, 0, 0, 0.34);
                margin-bottom: 1.5rem;
                overflow: hidden;
                padding: 2.15rem 2.35rem;
                position: relative;
            }

            .hero:after {
                background: rgba(255, 255, 255, 0.12);
                border-radius: 999px;
                content: "";
                height: 190px;
                position: absolute;
                right: -52px;
                top: -72px;
                width: 190px;
            }

            .hero-kicker {
                color: rgba(255, 255, 255, 0.80);
                font-size: 0.82rem;
                font-weight: 850;
                letter-spacing: 0.09em;
                margin-bottom: 0.6rem;
                text-transform: uppercase;
            }

            .hero-title {
                color: #ffffff;
                font-size: clamp(2.2rem, 4.5vw, 4rem);
                font-weight: 900;
                line-height: 1.02;
                margin: 0;
            }

            .hero-subtitle {
                color: rgba(255, 255, 255, 0.88);
                font-size: 1.08rem;
                line-height: 1.62;
                margin: 0.9rem 0 1.2rem;
                max-width: 860px;
            }

            .hero-chip {
                background: rgba(15, 23, 42, 0.28);
                border: 1px solid rgba(255, 255, 255, 0.16);
                border-radius: 999px;
                color: #ffffff;
                display: inline-block;
                font-size: 0.85rem;
                font-weight: 720;
                margin: 0.25rem 0.35rem 0 0;
                padding: 0.45rem 0.75rem;
            }

            .sidebar-title, .mini-card, .kpi-card, .prediction-card, .confidence-shell,
            .recommendation-card, .insight-card, .history-wrap, .chart-panel {
                border: 1px solid var(--border);
                box-shadow: 0 18px 42px rgba(0, 0, 0, 0.24);
            }

            .sidebar-title {
                background: rgba(56, 189, 248, 0.10);
                border-color: rgba(56, 189, 248, 0.24);
                border-radius: 18px;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
                padding: 1rem;
            }

            .sidebar-title h2 {
                color: #ffffff;
                font-size: 1.35rem;
                margin: 0;
            }

            .sidebar-title p {
                color: var(--muted);
                font-size: 0.9rem;
                margin: 0.3rem 0 0;
            }

            .section-label {
                color: #ffffff;
                font-size: 1.25rem;
                font-weight: 820;
                margin: 1.35rem 0 0.85rem;
            }

            .soft-divider {
                background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.55), transparent);
                height: 1px;
                margin: 1.65rem 0;
            }

            .kpi-card {
                border-radius: 18px;
                min-height: 150px;
                padding: 1.25rem;
                transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
            }

            .kpi-card:hover, .chart-panel:hover, .mini-card:hover {
                border-color: rgba(56, 189, 248, 0.42);
                box-shadow: 0 26px 52px rgba(0, 0, 0, 0.34);
                transform: translateY(-3px);
            }

            .kpi-blue { background: linear-gradient(135deg, rgba(56, 189, 248, 0.28), var(--surface)); }
            .kpi-green { background: linear-gradient(135deg, rgba(34, 197, 94, 0.28), var(--surface)); }
            .kpi-orange { background: linear-gradient(135deg, rgba(245, 158, 11, 0.28), var(--surface)); }
            .kpi-purple { background: linear-gradient(135deg, rgba(167, 139, 250, 0.28), var(--surface)); }

            .kpi-icon {
                font-size: 2rem;
                margin-bottom: 0.75rem;
            }

            .kpi-label {
                color: var(--muted);
                font-size: 0.9rem;
                font-weight: 760;
                margin-bottom: 0.35rem;
            }

            .kpi-value {
                color: #ffffff;
                font-size: 2.35rem;
                font-weight: 900;
                line-height: 1;
            }

            .kpi-delta {
                color: var(--muted);
                font-size: 0.83rem;
                margin-top: 0.78rem;
            }

            .prediction-card {
                align-items: center;
                animation: pulseIn 0.45s ease-out;
                border-color: rgba(255, 255, 255, 0.18);
                border-radius: 24px;
                box-shadow: 0 32px 82px rgba(0, 0, 0, 0.40);
                color: #ffffff;
                display: flex;
                flex-direction: column;
                justify-content: center;
                margin: 0.4rem auto 1rem;
                min-height: 290px;
                padding: 2.25rem;
                text-align: center;
                transition: transform 0.22s ease, box-shadow 0.22s ease;
            }

            .prediction-card:hover {
                box-shadow: 0 38px 92px rgba(0, 0, 0, 0.46);
                transform: translateY(-4px);
            }

            .prediction-good { background: linear-gradient(135deg, #047857, #22c55e 52%, #38bdf8); }
            .prediction-bad { background: linear-gradient(135deg, #7f1d1d, #ef4444 55%, #f97316); }

            .prediction-card h3 {
                color: #ffffff;
                font-size: 2.55rem;
                font-weight: 900;
                margin: 0 0 0.75rem;
            }

            .prediction-card p {
                color: #ffffff;
                font-size: 1.08rem;
                line-height: 1.6;
                margin: 0;
                max-width: 680px;
            }

            .confidence-shell, .recommendation-card, .insight-card, .history-wrap, .chart-panel, .mini-card {
                background: var(--glass);
                border-radius: 18px;
                transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
            }

            .confidence-shell, .recommendation-card {
                margin: 1rem auto 0;
                max-width: 820px;
                padding: 1.15rem 1.35rem;
            }

            .confidence-row {
                align-items: center;
                color: #ffffff;
                display: flex;
                font-weight: 820;
                justify-content: space-between;
                margin-bottom: 0.75rem;
            }

            .confidence-track {
                background: rgba(148, 163, 184, 0.18);
                border-radius: 999px;
                height: 16px;
                overflow: hidden;
            }

            .confidence-fill {
                background: linear-gradient(90deg, #38bdf8, #22c55e);
                border-radius: 999px;
                height: 100%;
                transition: width 0.5s ease;
            }

            .recommendation-card h4 {
                color: #ffffff;
                font-size: 1.35rem;
                margin: 0 0 0.55rem;
            }

            .recommendation-card p {
                color: #dbeafe;
                font-size: 1.05rem;
                line-height: 1.55;
                margin: 0;
            }

            .insight-card, .history-wrap, .chart-panel, .mini-card {
                padding: 1rem;
            }

            .mini-card {
                min-height: 126px;
            }

            .mini-title {
                color: #ffffff;
                font-weight: 820;
                margin-bottom: 0.35rem;
            }

            .mini-value {
                color: var(--blue);
                font-size: 1.35rem;
                font-weight: 850;
            }

            .stButton button {
                background: linear-gradient(135deg, #2563eb, #22c55e) !important;
                border: 0 !important;
                border-radius: 12px !important;
                box-shadow: 0 14px 32px rgba(37, 99, 235, 0.28) !important;
                color: #ffffff !important;
                font-weight: 800 !important;
                min-height: 2.85rem;
                transition: transform 0.2s ease, box-shadow 0.2s ease !important;
            }

            .stButton button:hover {
                box-shadow: 0 18px 38px rgba(34, 197, 94, 0.25) !important;
                transform: translateY(-2px);
            }

            [data-testid="stSelectbox"], [data-testid="stSlider"], [data-testid="stMultiSelect"] {
                background: rgba(255, 255, 255, 0.035);
                border: 1px solid rgba(148, 163, 184, 0.12);
                border-radius: 14px;
                padding: 0.35rem 0.45rem 0.6rem;
            }

            [data-testid="stTabs"] button {
                color: #dbeafe;
                font-weight: 760;
            }

            .streamlit-alert {
                border-radius: 14px;
                font-size: 1.08rem;
                margin: 0 auto 1rem;
                max-width: 820px;
            }

            @keyframes pulseIn {
                from { opacity: 0; transform: translateY(8px) scale(0.98); }
                to { opacity: 1; transform: translateY(0) scale(1); }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(icon: str, label: str, value: str, detail: str, color_class: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card {color_class}">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mini_card(title: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-title">{title}</div>
            <div class="mini-value">{value}</div>
            <p>{detail}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data["Day"] = pd.Categorical(data["Day"], categories=DAY_ORDER, ordered=True)
    data["Time"] = pd.Categorical(data["Time"], categories=TIME_ORDER, ordered=True)
    return data


@st.cache_resource
def train_model(data: pd.DataFrame):
    encoded_data = data.copy()
    encoders = {}

    for column in CATEGORY_COLUMNS:
        encoder = LabelEncoder()
        encoded_data[column] = encoder.fit_transform(encoded_data[column].astype(str))
        encoders[column] = encoder

    X = encoded_data[FEATURE_COLUMNS]
    y = encoded_data["Category"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, encoders


def encode_prediction_input(encoders, day, time_value, activity, mood, duration) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Day": encoders["Day"].transform([day])[0],
                "Time": encoders["Time"].transform([time_value])[0],
                "Activity": encoders["Activity"].transform([activity])[0],
                "Mood": encoders["Mood"].transform([mood])[0],
                "Duration": duration,
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def get_recommendation(category: str, activity: str, mood: str, duration: int) -> str:
    lower_activity = activity.lower()
    lower_mood = mood.lower()

    if category == "Good":
        if duration >= 3:
            return f"Keep {activity}, but add short breaks so the {duration}-hour session stays sustainable."
        if lower_mood in {"focused", "active", "calm", "excited"}:
            return f"Keep maintaining this habit. {activity} works especially well when you feel {mood}."
        return f"Continue {activity}, then note how your mood changes so you can repeat the best timing."

    if duration >= 3:
        return f"Reduce {activity} first. A {duration}-hour block is probably the biggest risk signal."
    if lower_activity in {"gaming", "ott", "social media"}:
        return f"Switch from {activity} to Reading, Study, or Exercise when you feel {mood}."
    if lower_mood in {"tired", "bored", "relaxed"}:
        return f"Since you feel {mood}, try a quick reset before continuing with {activity}."
    return f"Try changing the timing or replacing {activity} with a healthier option for this mood."


def get_prediction_message(category: str, confidence: float, activity: str, mood: str, duration: int) -> str:
    if category == "Good":
        if confidence >= 0.75:
            return f"{activity} for {duration} hour(s) with a {mood} mood strongly matches your productive patterns."
        return "This pattern looks productive and consistent, with moderate model confidence."

    if confidence >= 0.75:
        return f"{activity} for {duration} hour(s) strongly matches your weaker habit patterns."
    return "This pattern may negatively affect productivity. Treat it as a useful early warning."


def save_prediction_history(day, time_value, activity, mood, duration, category, confidence_percent) -> None:
    signature = (day, time_value, activity, mood, duration, category, confidence_percent)

    if st.session_state.last_prediction_signature == signature:
        return

    st.session_state.last_prediction_signature = signature
    st.session_state.prediction_history.insert(
        0,
        {
            "Day": day,
            "Time": time_value,
            "Activity": activity,
            "Mood": mood,
            "Duration": duration,
            "Prediction": category,
            "Confidence": f"{confidence_percent}%",
        },
    )
    st.session_state.prediction_history = st.session_state.prediction_history[:5]


def filter_data(data, days, times, activities, moods, duration_range) -> pd.DataFrame:
    return data[
        data["Day"].astype(str).isin(days)
        & data["Time"].astype(str).isin(times)
        & data["Activity"].isin(activities)
        & data["Mood"].isin(moods)
        & data["Duration"].between(duration_range[0], duration_range[1])
    ]


def rate_table(data, group_col) -> pd.DataFrame:
    grouped = (
        data.groupby(group_col, observed=True)
        .agg(
            Entries=("Category", "size"),
            Good=("Category", lambda values: int((values == "Good").sum())),
            AvgDuration=("Duration", "mean"),
        )
        .reset_index()
    )
    grouped["GoodRate"] = (grouped["Good"] / grouped["Entries"] * 100).round(1)
    return grouped.sort_values(["GoodRate", "Entries"], ascending=[False, False])


def style_chart(fig, height=390):
    fig.update_layout(
        dragmode="zoom",
        height=height,
        margin=dict(l=20, r=20, t=62, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        title_font=dict(color="#f8fafc", size=18),
    )
    return fig


apply_dashboard_css()

data = load_data()
model, encoders = train_model(data)

all_days = [day for day in DAY_ORDER if day in set(data["Day"].astype(str))]
all_times = [time_value for time_value in TIME_ORDER if time_value in set(data["Time"].astype(str))]
all_activities = sorted(data["Activity"].unique())
all_moods = sorted(data["Mood"].unique())

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-title">
            <h2>&#9881;&#65039; Controls</h2>
            <p>Predict one habit and filter the analytics live.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### &#129504; Prediction Studio", unsafe_allow_html=True)
    day = st.selectbox("&#128197; Day", all_days)
    selected_time = st.selectbox("&#9200; Time", all_times)
    activity = st.selectbox("&#127919; Activity", all_activities)
    mood = st.selectbox("&#128578; Mood", all_moods)
    duration = st.slider(
        "&#9201;&#65039; Duration (hours)",
        min_value=int(data["Duration"].min()),
        max_value=int(data["Duration"].max()),
        value=int(data["Duration"].median()),
    )

    st.markdown("#### &#128269; Analytics Filters", unsafe_allow_html=True)
    selected_days = st.multiselect("Days", all_days, default=all_days)
    selected_times = st.multiselect("Times", all_times, default=all_times)
    selected_activities = st.multiselect("Activities", all_activities, default=all_activities)
    selected_moods = st.multiselect("Moods", all_moods, default=all_moods)
    duration_range = st.slider(
        "Duration Range",
        min_value=int(data["Duration"].min()),
        max_value=int(data["Duration"].max()),
        value=(int(data["Duration"].min()), int(data["Duration"].max())),
    )

if not selected_days:
    selected_days = all_days
if not selected_times:
    selected_times = all_times
if not selected_activities:
    selected_activities = all_activities
if not selected_moods:
    selected_moods = all_moods

filtered_data = filter_data(data, selected_days, selected_times, selected_activities, selected_moods, duration_range)
analysis_data = filtered_data if not filtered_data.empty else data

total_entries = len(analysis_data)
good_count = int((analysis_data["Category"] == "Good").sum())
bad_count = int((analysis_data["Category"] == "Bad").sum())
good_rate = (good_count / total_entries * 100) if total_entries else 0
avg_duration = analysis_data["Duration"].mean() if total_entries else 0
risk_score = 100 - good_rate

activity_rates = rate_table(analysis_data, "Activity")
mood_rates = rate_table(analysis_data, "Mood")
best_activity = activity_rates.iloc[0]["Activity"] if not activity_rates.empty else "N/A"
risky_activity = activity_rates.sort_values(["GoodRate", "Entries"], ascending=[True, False]).iloc[0]["Activity"] if not activity_rates.empty else "N/A"
best_mood = mood_rates.iloc[0]["Mood"] if not mood_rates.empty else "N/A"
peak_time = analysis_data["Time"].astype(str).mode()[0] if total_entries else "N/A"

st.markdown(
    f"""
    <section class="hero">
        <div class="hero-kicker">&#128202; Interactive ML Habit Intelligence</div>
        <h1 class="hero-title">Habit Analyzer Pro</h1>
        <p class="hero-subtitle">
            A polished analytics dashboard for predicting habits, spotting patterns, and understanding what helps or hurts productivity.
        </p>
        <span class="hero-chip">{total_entries} filtered records</span>
        <span class="hero-chip">{good_rate:.0f}% good habit rate</span>
        <span class="hero-chip">Peak time: {peak_time}</span>
        <span class="hero-chip">Best mood: {best_mood}</span>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">&#128200; Performance Snapshot</div>', unsafe_allow_html=True)
kpi_col_1, kpi_col_2, kpi_col_3, kpi_col_4 = st.columns(4)
with kpi_col_1:
    kpi_card("&#128202;", "Filtered Entries", str(total_entries), f"{len(data)} total records available", "kpi-blue")
with kpi_col_2:
    kpi_card("&#128200;", "Good Habit Rate", f"{good_rate:.0f}%", f"{good_count} good vs {bad_count} bad", "kpi-green")
with kpi_col_3:
    kpi_card("&#9201;&#65039;", "Avg Duration", f"{avg_duration:.1f} hrs", f"{duration_range[0]}-{duration_range[1]} hr filter active", "kpi-orange")
with kpi_col_4:
    kpi_card("&#9888;&#65039;", "Risk Score", f"{risk_score:.0f}%", f"Riskiest: {risky_activity}", "kpi-purple")

st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">&#129504; Live Prediction Studio</div>', unsafe_allow_html=True)
try:
    with st.spinner("Analyzing habit pattern..."):
        time.sleep(0.2)
        sample = encode_prediction_input(encoders, day, selected_time, activity, mood, duration)
        prediction = model.predict(sample)[0]
        predicted_category = encoders["Category"].inverse_transform([prediction])[0]
        probabilities = model.predict_proba(sample)[0]

    confidence = float(probabilities[list(model.classes_).index(prediction)])
    confidence_percent = int(round(confidence * 100))
    prediction_message = get_prediction_message(predicted_category, confidence, activity, mood, duration)
    recommendation = get_recommendation(predicted_category, activity, mood, duration)
    save_prediction_history(day, selected_time, activity, mood, duration, predicted_category, confidence_percent)

    if predicted_category == "Good":
        prediction_class = "prediction-good"
        prediction_title = "GOOD HABIT"
        st.success("Good Habit")
    else:
        prediction_class = "prediction-bad"
        prediction_title = "BAD HABIT"
        st.error("Bad Habit")

    st.markdown(
        f"""
        <div class="prediction-card {prediction_class}">
            <h3>{prediction_title}</h3>
            <h3 style="font-size: 1.9rem; margin-bottom: 0.9rem;">Confidence: {confidence_percent}%</h3>
            <p>{prediction_message}</p>
        </div>
        <div class="confidence-shell">
            <div class="confidence-row">
                <span>Model Confidence</span>
                <span>{confidence_percent}%</span>
            </div>
            <div class="confidence-track">
                <div class="confidence-fill" style="width: {confidence_percent}%;"></div>
            </div>
        </div>
        <div class="recommendation-card">
            <h4>Recommendation</h4>
            <p>{recommendation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(confidence_percent)
except ValueError as error:
    st.error("Prediction failed")
    st.write(f"Please check the selected inputs. Details: {error}")

st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

tab_overview, tab_patterns, tab_explorer = st.tabs(["Overview", "Pattern Lab", "Data Explorer"])

with tab_overview:
    st.markdown('<div class="section-label">&#128161; Smart Insights</div>', unsafe_allow_html=True)
    insight_1, insight_2, insight_3, insight_4 = st.columns(4)
    with insight_1:
        mini_card("Best Activity", str(best_activity), "Highest good-habit rate in current filters.")
    with insight_2:
        mini_card("Risk Activity", str(risky_activity), "Lowest good-habit rate in current filters.")
    with insight_3:
        mini_card("Peak Time", str(peak_time), "Most frequent time slot in the filtered data.")
    with insight_4:
        mini_card("Best Mood", str(best_mood), "Mood most associated with good outcomes.")

    st.markdown('<div class="section-label">&#128202; Core Charts</div>', unsafe_allow_html=True)
    chart_col, pie_col = st.columns(2)

    with chart_col:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown("### Activity Distribution", unsafe_allow_html=True)
        activity_counts = analysis_data["Activity"].value_counts().rename_axis("Activity").reset_index(name="Count")
        fig = px.bar(
            activity_counts,
            x="Activity",
            y="Count",
            color="Activity",
            text="Count",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=["#38bdf8", "#22c55e", "#f59e0b", "#a78bfa", "#fb7185", "#2dd4bf"],
            title="Where your time is going",
        )
        fig.update_traces(textposition="outside", hovertemplate="<b>%{x}</b><br>Entries: %{y}<extra></extra>")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_chart(fig), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with pie_col:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown("### Habit Quality Split", unsafe_allow_html=True)
        category_counts = analysis_data["Category"].value_counts().rename_axis("Category").reset_index(name="Count")
        fig = px.pie(
            category_counts,
            names="Category",
            values="Count",
            hole=0.52,
            color="Category",
            color_discrete_map={"Good": "#22c55e", "Bad": "#ef4444"},
            template=PLOTLY_TEMPLATE,
            title="Good vs bad habits",
        )
        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
            textinfo="percent+label",
        )
        st.plotly_chart(style_chart(fig), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    heat_col, mood_col = st.columns(2)
    with heat_col:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown("### Day x Time Heatmap", unsafe_allow_html=True)
        heatmap_data = (
            analysis_data.assign(Day=analysis_data["Day"].astype(str), Time=analysis_data["Time"].astype(str))
            .pivot_table(index="Day", columns="Time", values="Category", aggfunc="count", fill_value=0)
            .reindex(index=DAY_ORDER, columns=TIME_ORDER)
            .fillna(0)
        )
        fig = px.imshow(
            heatmap_data,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=["#111827", "#2563eb", "#22c55e"],
            template=PLOTLY_TEMPLATE,
            title="Habit density by schedule",
        )
        st.plotly_chart(style_chart(fig), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with mood_col:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown("### Mood Impact", unsafe_allow_html=True)
        mood_chart_data = rate_table(analysis_data, "Mood")
        fig = px.bar(
            mood_chart_data,
            x="Mood",
            y="GoodRate",
            color="GoodRate",
            text="GoodRate",
            color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
            template=PLOTLY_TEMPLATE,
            title="Good habit rate by mood",
        )
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        st.plotly_chart(style_chart(fig), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

with tab_patterns:
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown("### Activity Quality Ranking", unsafe_allow_html=True)
        fig = px.bar(
            activity_rates,
            x="GoodRate",
            y="Activity",
            color="GoodRate",
            orientation="h",
            text="GoodRate",
            color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
            template=PLOTLY_TEMPLATE,
            title="Which activities help most",
        )
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        st.plotly_chart(style_chart(fig, 430), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown("### Duration by Category", unsafe_allow_html=True)
        fig = px.box(
            analysis_data,
            x="Category",
            y="Duration",
            color="Category",
            points="all",
            color_discrete_map={"Good": "#22c55e", "Bad": "#ef4444"},
            template=PLOTLY_TEMPLATE,
            title="How long good and bad habits last",
        )
        st.plotly_chart(style_chart(fig, 430), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    stacked_col, scatter_col = st.columns(2)
    with stacked_col:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown("### Weekly Quality Mix", unsafe_allow_html=True)
        weekly_mix = (
            analysis_data.assign(Day=analysis_data["Day"].astype(str))
            .groupby(["Day", "Category"], observed=True)
            .size()
            .reset_index(name="Count")
        )
        fig = px.bar(
            weekly_mix,
            x="Day",
            y="Count",
            color="Category",
            category_orders={"Day": DAY_ORDER},
            color_discrete_map={"Good": "#22c55e", "Bad": "#ef4444"},
            template=PLOTLY_TEMPLATE,
            title="Good and bad habits across the week",
        )
        st.plotly_chart(style_chart(fig), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with scatter_col:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown("### Habit Map", unsafe_allow_html=True)
        scatter_data = analysis_data.copy()
        scatter_data["GoodFlag"] = (scatter_data["Category"] == "Good").astype(int)
        fig = px.scatter(
            scatter_data,
            x="Duration",
            y="Activity",
            color="Category",
            size="GoodFlag",
            hover_data=["Day", "Time", "Mood"],
            color_discrete_map={"Good": "#22c55e", "Bad": "#ef4444"},
            template=PLOTLY_TEMPLATE,
            title="Activities by duration and outcome",
        )
        st.plotly_chart(style_chart(fig), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

with tab_explorer:
    st.markdown('<div class="section-label">&#128269; Searchable Data Explorer</div>', unsafe_allow_html=True)
    search = st.text_input("Search activity, mood, day, time, or category", "")
    explorer_data = analysis_data.copy()
    if search:
        search_lower = search.lower()
        explorer_data = explorer_data[
            explorer_data.astype(str).apply(lambda row: row.str.lower().str.contains(search_lower).any(), axis=1)
        ]

    table_col, rank_col = st.columns([1.4, 1])
    with table_col:
        st.markdown('<div class="history-wrap">', unsafe_allow_html=True)
        st.dataframe(explorer_data, hide_index=True, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with rank_col:
        st.markdown('<div class="history-wrap">', unsafe_allow_html=True)
        st.markdown("### Recent Predictions")
        if st.session_state.prediction_history:
            st.dataframe(pd.DataFrame(st.session_state.prediction_history), hide_index=True, width="stretch")
        else:
            st.info("Change prediction controls to build history.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="history-wrap" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown("### Activity Scorecard")
        st.dataframe(activity_rates, hide_index=True, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
