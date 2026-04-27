import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import agent
from models.risk_model import RiskModel


st.set_page_config(page_title="Healthcare AI Agent", page_icon=":hospital:", layout="wide")

st.title("Healthcare AI Agent")
st.caption("Educational prototype only. Not medical advice.")

with st.sidebar:
    st.header("Patient")
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.5)
    weight_kg = st.number_input("Weight (kg)", min_value=2.0, max_value=300.0, value=82.4, step=0.1)
    bp = st.number_input("Blood Pressure", min_value=0, max_value=260, value=130)
    has_glucose = st.checkbox("I have a glucose value", value=True)
    glucose = None

    if has_glucose:
        glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=500, value=180)
    else:
        st.info("Without a glucose reading, the model uses a training-data median. Treat the result as lower confidence.")

    analyze = st.button("Analyze", type="primary", use_container_width=True)

if analyze:
    calculated_bmi = RiskModel.calculate_bmi(height_cm, weight_kg)
    patient = {
        "age": age,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bmi": round(calculated_bmi, 1),
        "bp": bp,
        "glucose": glucose,
    }
    st.session_state["last_result"] = agent.run(patient)

result = st.session_state.get("last_result")

if result:
    patient = result["patient"]
    safety = result.get("safety", {})
    confidence = safety.get("confidence_label", "Unknown")

    metric_cols = st.columns(5)
    metric_cols[0].metric("Risk", result["risk"])
    metric_cols[1].metric("Probability", f"{result['probability']:.1%}")
    metric_cols[2].metric("Calculated BMI", f"{patient['bmi']:.1f}")
    metric_cols[3].metric("Confidence", confidence)
    metric_cols[4].metric("Escalation", safety.get("escalation", "routine_followup"))

    if safety.get("alerts"):
        for alert in safety["alerts"]:
            st.warning(alert)
    for disclaimer in safety.get("disclaimers", []):
        st.caption(disclaimer)

    dashboard_tab, memory_tab, knowledge_tab = st.tabs(["Dashboard", "Memory", "Knowledge"])

    with dashboard_tab:
        left, right = st.columns([1.2, 1])

        with left:
            st.subheader("Explanation")
            st.write(result["explanation"])

            st.subheader("Recommendation")
            st.write(result["recommendation"])

        with right:
            st.subheader("Feature Impact")
            features = result["feature_explanation"]["features"]
            if features:
                impact_df = pd.DataFrame(features)
                chart_df = impact_df[["feature", "impact"]].set_index("feature")
                st.bar_chart(chart_df)
                st.dataframe(
                    impact_df[["feature", "value", "direction", "impact"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.write("No feature explanation available.")

    with memory_tab:
        st.subheader("Similar Cases")
        if result["similar_cases"]:
            st.dataframe(
                pd.DataFrame([
                    {
                        "risk": item["metadata"].get("risk"),
                        "probability": item["metadata"].get("probability"),
                        "age": item["metadata"].get("age"),
                        "bmi": item["metadata"].get("bmi"),
                        "bp": item["metadata"].get("bp"),
                        "glucose": item["metadata"].get("glucose"),
                        "distance": item["distance"],
                    }
                    for item in result["similar_cases"]
                ]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No prior cases stored yet.")

        st.subheader("Recent Memory")
        history = agent.memory.get_all()
        if history:
            st.dataframe(
                pd.DataFrame([
                    {
                        "timestamp": item["metadata"].get("timestamp"),
                        "risk": item["metadata"].get("risk"),
                        "probability": item["metadata"].get("probability"),
                        "age": item["metadata"].get("age"),
                        "bmi": item["metadata"].get("bmi"),
                        "bp": item["metadata"].get("bp"),
                        "glucose": item["metadata"].get("glucose"),
                    }
                    for item in history
                ]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No memory records yet.")

    with knowledge_tab:
        st.subheader("Retrieved Medical Context")
        for item in result["retrieved_context"]:
            with st.expander(item["title"], expanded=True):
                st.write(item["text"])
else:
    st.info("Enter patient values in the sidebar and run an analysis.")
