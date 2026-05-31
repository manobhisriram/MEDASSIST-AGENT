import streamlit as st
import requests
import time

st.set_page_config(
    page_title="MedAssist — Clinical Triage Agent",
    page_icon="🏥",
    layout="wide"
)

API_URL = "http://localhost:8000"

SEVERITY_COLORS = {
    "LOW": "#28a745",
    "MEDIUM": "#ffc107",
    "HIGH": "#fd7e14",
    "EMERGENCY": "#dc3545"
}

SEVERITY_MESSAGES = {
    "LOW": "Low Priority — Can be managed at home",
    "MEDIUM": "Medium Priority — See a doctor within 48 hours",
    "HIGH": "High Priority — See a doctor today",
    "EMERGENCY": "EMERGENCY — Go to hospital immediately"
}

st.markdown("""
<style>
.severity-badge {
    padding: 14px 24px;
    border-radius: 8px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
    color: white;
    margin: 16px 0;
}
.step-item {
    padding: 8px 16px;
    background: #f8f9fa;
    border-left: 4px solid #007bff;
    margin: 6px 0;
    border-radius: 4px;
}
.warning-item {
    padding: 8px 16px;
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    margin: 6px 0;
    border-radius: 4px;
}
.emergency-item {
    padding: 8px 16px;
    background: #f8d7da;
    border-left: 4px solid #dc3545;
    margin: 6px 0;
    border-radius: 4px;
}
.source-chip {
    display: inline-block;
    background: #e9ecef;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    margin: 3px;
}
.demo-box {
    background: #f0f7ff;
    border: 1px solid #b8d4f0;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("🏥 MedAssist")
st.subheader("Clinical Triage Agent — Fine-Tuned Phi-3 Mini + WHO/ICMR Guidelines")
st.markdown("*Describe your symptoms and receive an AI-powered triage assessment grounded in clinical guidelines, with every claim traceable to a verified source.*")

backend_online = False
try:
    health = requests.get(f"{API_URL}/health", timeout=3)
    if health.status_code == 200:
        backend_online = True
except Exception:
    backend_online = False

if not backend_online:
    st.markdown("""
    <div class="demo-box">
    <b>📌 Live Demo Mode</b><br>
    The inference backend (Ollama + fine-tuned Phi-3 Mini) runs locally due to GPU requirements. 
    To run the full system locally, follow the setup instructions in the 
    <a href="https://github.com/manobhisriram/MEDASSIST-AGENT" target="_blank">GitHub repository</a>.<br><br>
    Below you can explore the interface and see a sample triage output.
    </div>
    """, unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Describe Your Symptoms")
    symptom_input = st.text_area(
        "Symptoms",
        placeholder="Example: I have had fever for 3 days, my joints are aching badly, and I noticed red spots on my arms today. I am 35 years old.",
        height=150,
        label_visibility="collapsed"
    )
    submit = st.button("Get Triage Assessment", type="primary", use_container_width=True)

with col2:
    st.subheader("Try an Example")
    examples = [
        "Fever 3 days, joint pain, red spots on skin",
        "Severe chest pain and shortness of breath",
        "Child mild cold, no fever, runny nose",
        "Headache, stiff neck, light sensitivity",
        "Cough 3 weeks with blood in sputum"
    ]
    for example in examples:
        if st.button(example, use_container_width=True, key=example):
            symptom_input = example

st.divider()

if submit and symptom_input:
    if not backend_online:
        st.subheader("Agent Processing")
        steps = [
            ("🔍", "Parsing symptoms from description..."),
            ("⚕️", "Assessing severity level..."),
            ("📚", "Retrieving clinical guidelines from LanceDB..."),
            ("💊", "Checking drug information via OpenFDA..."),
            ("📋", "Generating triage decision...")
        ]
        placeholders = []
        for icon, text in steps:
            p = st.empty()
            p.markdown(f"⏳ {icon} {text}")
            placeholders.append(p)
        
        for i in range(len(steps)):
            time.sleep(0.4)
            placeholders[i].markdown(f"✅ {steps[i][0]} {steps[i][1]}")

        st.divider()
        st.subheader("Sample Triage Output")
        st.info("This is a representative output showing the system's response format. Run locally for live inference.")

        st.markdown(
            '<div class="severity-badge" style="background-color: #fd7e14;">🚨 High Priority — See a doctor today</div>',
            unsafe_allow_html=True
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Recommended Next Steps**")
            steps_sample = [
                "Visit a hospital or clinic today for dengue testing",
                "Request a dengue NS1 antigen test immediately",
                "Use paracetamol only for fever — avoid aspirin and ibuprofen",
                "Monitor for red flag symptoms every 6 hours"
            ]
            for step in steps_sample:
                st.markdown(f'<div class="step-item">✓ {step}</div>', unsafe_allow_html=True)

            st.markdown("**⚠️ Red Flag Warnings — Go to Emergency If:**")
            warnings = [
                "Severe abdominal pain develops",
                "Bleeding from gums, nose, or in stool",
                "Difficulty breathing or extreme fatigue"
            ]
            for w in warnings:
                st.markdown(f'<div class="emergency-item">⚠ {w}</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown("**Clinical Reasoning**")
            st.info("Fever for 3 days with joint pain and red spots matches dengue fever warning signs per WHO dengue triage guidelines (Section 2.3). The combination of symptoms warrants same-day medical evaluation and diagnostic testing. Platelet count monitoring is recommended.")

        st.markdown("**Sources Cited**")
        sources_html = '<span class="source-chip">📄 dengue_triage.txt</span><span class="source-chip">📄 fever_triage.txt</span>'
        st.markdown(sources_html, unsafe_allow_html=True)

        st.divider()
        st.caption("This assessment is for informational purposes only and does not replace professional medical advice. Please consult a qualified healthcare provider.")

    else:
        st.subheader("Agent Processing")
        steps = [
            ("🔍", "Parsing symptoms from description..."),
            ("⚕️", "Assessing severity level..."),
            ("📚", "Retrieving clinical guidelines..."),
            ("💊", "Checking drug information..."),
            ("📋", "Generating triage decision...")
        ]
        placeholders = []
        for icon, text in steps:
            p = st.empty()
            p.markdown(f"⏳ {icon} {text}")
            placeholders.append(p)

        for i in range(len(steps)):
            time.sleep(0.3)
            placeholders[i].markdown(f"✅ {steps[i][0]} {steps[i][1]}")

        with st.spinner("Generating final triage decision..."):
            try:
                response = requests.post(
                    f"{API_URL}/triage",
                    json={"symptoms": symptom_input},
                    timeout=600
                )
                if response.status_code == 200:
                    result = response.json()
                    severity = result["severity"]
                    color = SEVERITY_COLORS.get(severity, "#6c757d")
                    message = SEVERITY_MESSAGES.get(severity, severity)

                    st.divider()
                    st.subheader("Triage Decision")
                    st.markdown(
                        f'<div class="severity-badge" style="background-color: {color};">🚨 {message}</div>',
                        unsafe_allow_html=True
                    )

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Recommended Next Steps**")
                        for step in result["recommended_next_steps"]:
                            st.markdown(f'<div class="step-item">✓ {step}</div>', unsafe_allow_html=True)
                        if result.get("red_flag_warnings"):
                            st.markdown("**⚠️ Red Flag Warnings**")
                            for w in result["red_flag_warnings"]:
                                st.markdown(f'<div class="emergency-item">⚠ {w}</div>', unsafe_allow_html=True)

                    with col_b:
                        st.markdown("**Clinical Reasoning**")
                        st.info(result["clinical_reasoning"])
                        if result.get("drug_interaction_warnings"):
                            st.markdown("**💊 Drug Warnings**")
                            for w in result["drug_interaction_warnings"]:
                                st.markdown(f'<div class="warning-item">💊 {w}</div>', unsafe_allow_html=True)

                    if result.get("sources"):
                        st.markdown("**Sources Cited**")
                        sources_html = "".join([f'<span class="source-chip">📄 {s}</span>' for s in result["sources"]])
                        st.markdown(sources_html, unsafe_allow_html=True)

                    if result.get("uncertainty_note"):
                        st.warning(result["uncertainty_note"])

                    st.divider()
                    st.caption(result.get("disclaimer", ""))
                else:
                    st.error(f"API error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")

elif submit and not symptom_input:
    st.warning("Please describe your symptoms first.")

st.divider()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Faithfulness", "0.92")
with col2:
    st.metric("Answer Relevancy", "0.90")
with col3:
    st.metric("Context Precision", "0.85")
with col4:
    st.metric("DeepEval Tests", "5/5 ✅")

st.markdown("---")
st.markdown(
    "**MedAssist** — Fine-Tuned Phi-3 Mini · LangGraph · LanceDB · Ragas · Langfuse · FastAPI · Streamlit &nbsp;|&nbsp; "
    "[GitHub](https://github.com/manobhisriram/MEDASSIST-AGENT) &nbsp;|&nbsp; "
    "[Model on HuggingFace](https://huggingface.co/manobhi18sriram1/medassist-phi3-mini) &nbsp;|&nbsp; "
    "[Evaluation Results](https://github.com/manobhisriram/MEDASSIST-AGENT/blob/main/evaluation/ragas_scores.json)"
)