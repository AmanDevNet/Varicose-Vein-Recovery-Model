import os
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st

from varicose_vein_model import VaricoseVeinMLModel


def load_local_env(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a local .env file into process env if present."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


load_local_env()

st.set_page_config(
    page_title="Varicose Vein Recovery Assistant",
    layout="wide",
)


def format_narrative_source(source: str) -> str:
    mapping = {
        "gemini": "Gemini AI",
        "fallback_no_key": "Rule-based (Gemini key not loaded)",
        "fallback_model_not_found": "Rule-based (configured Gemini model not available)",
        "fallback_requests_missing": "Rule-based (requests package missing)",
        "fallback_api_error": "Rule-based (Gemini API error)",
        "fallback_empty_response": "Rule-based (empty Gemini response)",
        "rule_based": "Rule-based",
    }
    return mapping.get(source, "Rule-based")


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap');

        :root {
            --ink: #24363d;
            --ink-soft: #5b6d74;
            --panel: #f2f4f5;
            --panel-border: #c6d1d5;
            --page: #edf0f1;
        }

        html, body, [class*="css"], .stApp {
            font-family: 'Manrope', sans-serif;
            color: var(--ink) !important;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 10%, rgba(209, 223, 226, 0.5), transparent 40%),
                radial-gradient(circle at 88% 14%, rgba(224, 229, 231, 0.45), transparent 36%),
                var(--page);
        }

        .block-container {
            max-width: 1100px;
            padding-top: 1.2rem;
            padding-bottom: 1.5rem;
        }

        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp p, .stApp li, .stApp label, .stApp span, .stApp small {
            color: var(--ink) !important;
        }

        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 12px 14px;
            box-shadow: 0 5px 16px rgba(34, 49, 56, 0.06);
            animation: riseIn 0.35s ease-out;
        }

        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
            color: var(--ink-soft) !important;
            font-weight: 600 !important;
        }

        [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
            color: var(--ink) !important;
        }

        [data-testid="stSidebar"] {
            background: #e8edef;
            border-right: 1px solid #cad4d8;
        }

        [data-testid="stSidebar"] * {
            color: var(--ink) !important;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stMultiSelect"] > div > div {
            background: #f7f9f9 !important;
            color: var(--ink) !important;
            border-color: #c8d3d7 !important;
        }

        .helper-text {
            color: var(--ink-soft) !important;
            font-size: 0.86rem;
        }

        @keyframes riseIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model() -> VaricoseVeinMLModel:
    model = VaricoseVeinMLModel()
    try:
        if model.artifacts_exist():
            model.load_artifacts()
        else:
            data = model.generate_synthetic_data(6000)
            model.train_models(
                data=data,
                enable_tuning=False,
                save_artifacts=False,
                verbose=False,
                compute_cv=False,
            )
    except Exception:
        data = model.generate_synthetic_data(6000)
        model.train_models(
            data=data,
            enable_tuning=False,
            save_artifacts=False,
            verbose=False,
            compute_cv=False,
        )
    return model


def build_sidebar_form() -> tuple[List[float], bool, bool]:
    with st.sidebar:
        st.title("Patient Input")
        st.caption("Use realistic values for best prediction quality.")

        with st.form("prediction_form", clear_on_submit=False):
            age = st.slider("Age", 20, 80, 48)
            bmi = st.slider("BMI", 15.0, 45.0, 26.7, step=0.1)
            family_history = st.selectbox("Family history of varicose veins", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            pain_level = st.slider("Pain level", 0.0, 10.0, 7.0, step=0.1)
            swelling = st.slider("Swelling level", 0.0, 10.0, 6.0, step=0.1)
            activity_level = st.slider("Activity level", 0.0, 10.0, 8.0, step=0.1)

            st.markdown("### Supplements")
            beetroot_days_30 = st.slider("Beetroot intake days (last 30)", 0, 30, 25)
            beetroot_grams_day = st.slider("Beetroot grams/day", 0.0, 30.0, 11.0, step=0.1)
            fenugreek_days_30 = st.slider("Fenugreek intake days (last 30)", 0, 30, 25)
            fenugreek_grams_day = st.slider("Fenugreek grams/day", 0.0, 20.0, 15.0, step=0.1)

            st.markdown("### AI Narrative")
            use_ai = st.checkbox("Use Gemini response", value=False)
            has_key = bool(os.getenv("GEMINI_API_KEY", "").strip())
            if has_key:
                st.caption("Gemini key detected from environment.")
            else:
                st.caption("No Gemini key found. AI response will use fallback narrative.")
            submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    user_input = [
        float(age),
        float(bmi),
        float(family_history),
        float(pain_level),
        float(swelling),
        float(activity_level),
        float(beetroot_days_30),
        float(beetroot_grams_day),
        float(fenugreek_days_30),
        float(fenugreek_grams_day),
    ]
    return user_input, submitted, use_ai


def main() -> None:
    apply_styles()
    user_input, submitted, use_ai = build_sidebar_form()

    st.title("Varicose Vein Recovery Assistant")
    st.markdown(
        '<p class="helper-text">Minimal clinical-style dashboard for risk and recovery estimation. '
        "Predictions are model-generated and should be used as decision support only.</p>",
        unsafe_allow_html=True,
    )

    if not submitted:
        st.info("Set inputs in the sidebar and click Run Prediction.")
        return

    with st.spinner("Loading model..."):
        try:
            model = load_model()
        except Exception as exc:
            st.error("Failed to load or train the model.")
            st.exception(exc)
            return

    risk_level, recovery_weeks, risk_probs = model.predict_single_case(user_input, return_probabilities=True)
    suggestions = model.generate_suggestions(user_input, risk_level, recovery_weeks)
    comparison = model.compare_supplement_effect(user_input)

    if use_ai:
        narrative_text, narrative_source = model.generate_ai_narrative(
            user_input=user_input,
            risk_level=risk_level,
            recovery_weeks=recovery_weeks,
            suggestions=suggestions,
            comparison=comparison,
        )
    else:
        narrative_text = model.generate_rule_based_narrative(
            user_input=user_input,
            risk_level=risk_level,
            recovery_weeks=recovery_weeks,
            suggestions=suggestions,
            comparison=comparison,
        )
        narrative_source = "rule_based"

    gain_vs_none = comparison["no_supplements"]["recovery_weeks"] - comparison["current"]["recovery_weeks"]
    current_vs_opt = comparison["current"]["recovery_weeks"] - comparison["optimal_supplements"]["recovery_weeks"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Level", risk_level)
    c2.metric("Estimated Recovery", f"{recovery_weeks:.1f} weeks")
    c3.metric("Gain vs No Supplements", f"{gain_vs_none:.1f} weeks")

    c4, c5, c6 = st.columns(3)
    c4.metric("Current vs Optimized Gap", f"{current_vs_opt:.1f} weeks")
    c5.metric("Beetroot Monthly Dose", f"{user_input[6] * user_input[7]:.0f} g")
    c6.metric("Fenugreek Monthly Dose", f"{user_input[8] * user_input[9]:.0f} g")

    prob_df = pd.DataFrame(
        {
            "Risk Level": ["Low", "Moderate", "High"],
            "Probability": [
                risk_probs.get("Low", 0.0),
                risk_probs.get("Moderate", 0.0),
                risk_probs.get("High", 0.0),
            ],
        }
    )
    prob_fig = px.bar(
        prob_df,
        x="Risk Level",
        y="Probability",
        color="Risk Level",
        text="Probability",
        color_discrete_map={"Low": "#2D6A4F", "Moderate": "#E9C46A", "High": "#D1495B"},
    )
    prob_fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    prob_fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        template="plotly_white",
        paper_bgcolor="#f2f4f5",
        plot_bgcolor="#f2f4f5",
        font=dict(color="#24363d"),
    )
    prob_fig.update_yaxes(range=[0, 1], tickformat=".0%")

    comp_df = pd.DataFrame(
        [
            {
                "Scenario": "Current",
                "Recovery Weeks": comparison["current"]["recovery_weeks"],
                "Risk": comparison["current"]["risk"],
            },
            {
                "Scenario": "No Supplements",
                "Recovery Weeks": comparison["no_supplements"]["recovery_weeks"],
                "Risk": comparison["no_supplements"]["risk"],
            },
            {
                "Scenario": "Optimized",
                "Recovery Weeks": comparison["optimal_supplements"]["recovery_weeks"],
                "Risk": comparison["optimal_supplements"]["risk"],
            },
        ]
    )
    comp_fig = px.bar(
        comp_df,
        x="Scenario",
        y="Recovery Weeks",
        color="Risk",
        text="Recovery Weeks",
        color_discrete_map={"Low": "#2D6A4F", "Moderate": "#E9C46A", "High": "#D1495B"},
    )
    comp_fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    comp_fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
        paper_bgcolor="#f2f4f5",
        plot_bgcolor="#f2f4f5",
        font=dict(color="#24363d"),
    )

    left, right = st.columns(2)
    left.plotly_chart(prob_fig, use_container_width=True)
    right.plotly_chart(comp_fig, use_container_width=True)

    trajectory = model.build_recovery_trajectory(user_input, recovery_weeks)
    trajectory_long = trajectory.melt(id_vars="week", var_name="Metric", value_name="Score")
    trajectory_long["Metric"] = trajectory_long["Metric"].str.capitalize()
    trend_fig = px.line(
        trajectory_long,
        x="week",
        y="Score",
        color="Metric",
        markers=True,
        color_discrete_map={"Pain": "#C0392B", "Swelling": "#E67E22", "Activity": "#1E8449"},
    )
    trend_fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
        paper_bgcolor="#f2f4f5",
        plot_bgcolor="#f2f4f5",
        font=dict(color="#24363d"),
    )

    intake = model.build_intake_impact(user_input)
    intake_long = intake.melt(
        id_vars="day",
        value_vars=["beetroot_effect", "fenugreek_effect", "combined_effect"],
        var_name="Metric",
        value_name="Effect",
    )
    intake_long["Metric"] = intake_long["Metric"].map(
        {
            "beetroot_effect": "Beetroot",
            "fenugreek_effect": "Fenugreek",
            "combined_effect": "Combined",
        }
    )
    intake_fig = px.area(
        intake_long,
        x="day",
        y="Effect",
        color="Metric",
        color_discrete_map={"Beetroot": "#B03A2E", "Fenugreek": "#196F3D", "Combined": "#547E7A"},
    )
    intake_fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
        paper_bgcolor="#f2f4f5",
        plot_bgcolor="#f2f4f5",
        font=dict(color="#24363d"),
    )

    lower_left, lower_right = st.columns(2)
    lower_left.plotly_chart(trend_fig, use_container_width=True)
    lower_right.plotly_chart(intake_fig, use_container_width=True)

    st.subheader("Suggested Actions")
    for idx, item in enumerate(suggestions, start=1):
        st.write(f"{idx}. {item}")

    st.subheader("Narrative")
    st.caption(f"Source: {format_narrative_source(narrative_source)}")
    st.write(narrative_text)

    summary_text = model.build_results_summary_text(
        user_input=user_input,
        risk_level=risk_level,
        recovery_weeks=recovery_weeks,
        suggestions=suggestions,
        comparison=comparison,
        narrative_text=narrative_text,
        narrative_source=narrative_source,
    )
    st.download_button(
        "Download Summary (TXT)",
        data=summary_text,
        file_name="results_summary.txt",
        mime="text/plain",
    )


if __name__ == "__main__":
    main()
