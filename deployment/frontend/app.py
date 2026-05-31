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
</style>
""", unsafe_allow_html=True)

st.title("🏥 MedAssist")
st.subheader("Clinical Triage Agent — Powered by Fine-Tuned Phi-3 Mini + Clinical Guidelines")
st.markdown("*Describe your symptoms and get an AI-powered triage assessment grounded in WHO and ICMR clinical guidelines.*")
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
                st.error(f"API error: {response.status_code} — {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to MedAssist API. Make sure the backend is running on port 8000.")
        except Exception as e:
            st.error(f"Error: {e}")

elif submit and not symptom_input:
    st.warning("Please describe your symptoms first.")

st.divider()

try:
    metrics = requests.get(f"{API_URL}/metrics", timeout=3).json()
    m1, m2, m3 = st.columns(3)
    scores = metrics.get("ragas_scores", {})
    with m1:
        st.metric("Faithfulness", f"{scores.get('faithfulness', 0):.2f}" if scores else "N/A")
    with m2:
        st.metric("Total Runs", metrics.get("total_runs", 0))
    with m3:
        st.metric("Model", "Phi-3 Mini (local)")
except Exception:
    pass

st.markdown("---")
st.markdown(
    "Built with Phi-3 Mini · LangGraph · LanceDB · Langfuse · FastAPI · Streamlit | "
    "[GitHub](https://github.com/manobhisriram/MEDASSIST-AGENT) | "
    "[Model](https://huggingface.co/manobhi18sriram1/medassist-phi3-mini)"
)