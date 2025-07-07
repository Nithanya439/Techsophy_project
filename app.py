import os
import streamlit as st
from transformers import pipeline

# ========== Load Model ==========
@st.cache_resource
def load_model():
    if os.path.exists(r"C:\\"):
        os.environ['TRANSFORMERS_CACHE'] = r"C:\\transformers_cache"
        os.makedirs(r"C:\\transformers_cache", exist_ok=True)
    return pipeline("text-generation", model="gpt2-medium")


# ========== Generate Prompt ==========
def get_medication_info(medications, patient_info):
    med_list = ", ".join(medications)
    prompt = (
        f"Patient Profile:\n"
        f"- Age: {patient_info['age']}\n"
        f"- Weight: {patient_info['weight']} kg\n"
        f"- Conditions: {', '.join(patient_info['conditions'])}\n"
        f"- Renal Impairment: {'Yes' if patient_info['renal_impairment'] else 'No'}\n"
        f"- Liver Problems: {'Yes' if patient_info['liver_problems'] else 'No'}\n\n"
        f"Medications:\n{med_list}\n\n"
        "Potential drug interactions, safety considerations, and alternative suggestions:"
    )
    return prompt


# ========== Generate AI Response ==========
def generate_response(model, prompt):
    response = model(
        prompt,
        max_length=600,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )
    return response[0]['generated_text'].replace(prompt, "").strip()


# ========== Page Config ==========
st.set_page_config(
    page_title="MedSafe AI - Medication Interaction Checker",
    page_icon="💊",
    layout="wide"
)

# ========== Load Model ==========
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()


# ========== Sidebar Input ==========
with st.sidebar:
    st.header("Patient Information")
    age = st.slider("Age", 1, 120, 45)
    weight = st.number_input("Weight (kg)", 1, 200, 70)
    conditions = st.text_input("Medical Conditions (comma separated)", "Hypertension, Diabetes")
    renal_impairment = st.checkbox("Renal Impairment")
    liver_problems = st.checkbox("Liver Problems")
    st.divider()
    st.info("""Disclaimer:\nThis AI-powered tool provides preliminary information only.\nAlways consult a healthcare professional.""")


# ========== Main UI ==========
st.title("💊 MedSafe AI - Medication Interaction Checker")
st.subheader("AI-powered medication safety analysis")

with st.form("med_input"):
    meds = st.text_area("Enter Medications (one per line or comma separated)", "Warfarin\nIbuprofen\nLisinopril", height=150)
    analyze_btn = st.form_submit_button("Analyze Medications", type="primary")


# ========== Processing ==========
if analyze_btn:
    medications = [m.strip() for m in meds.replace(',', '\n').split('\n') if m.strip()]
    patient_info = {
        'age': age,
        'weight': weight,
        'conditions': [c.strip() for c in conditions.split(',')],
        'renal_impairment': renal_impairment,
        'liver_problems': liver_problems
    }

    if not medications:
        st.warning("Please enter at least one medication")
        st.stop()

    with st.spinner("Analyzing potential interactions..."):
        try:
            prompt = get_medication_info(medications, patient_info)
            response = generate_response(model, prompt)

            st.subheader("Analysis Results")
            st.info(f"Analyzed {len(medications)} medications for patient ({age}y, {weight}kg)")

            with st.expander("Medication List", expanded=True):
                cols = st.columns(3)
                for i, med in enumerate(medications):
                    cols[i % 3].write(f"- {med}")

            st.subheader("Interaction Analysis")
            st.markdown(
                f"<div style='background-color: #0E1117; padding: 20px; border-radius: 10px; color:white;'>{response}</div>",
                unsafe_allow_html=True
            )

            st.error("""
            Important Considerations:
            - This analysis is AI-generated and may contain inaccuracies
            - Always verify with professional medical references
            - Dosage adjustments may be needed based on clinical context
            """)
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
else:
    st.markdown("""
    <style>
    .instruction-box {
        background-color: #0E1117;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="instruction-box">
        <h3>How to Use:</h3>
        <ol>
            <li>Enter patient information in the sidebar</li>
            <li>List medications (one per line or comma separated)</li>
            <li>Click "Analyze Medications"</li>
        </ol>

        <h3>Example Input:</h3>
        <pre>Warfarin
Ibuprofen
Lisinopril
Simvastatin</pre>

        <h3>What This Tool Analyzes:</h3>
        <ul>
            <li>Potential drug-drug interactions</li>
            <li>Patient-specific risk factors</li>
            <li>Possible alternative medications</li>
            <li>Special population considerations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("Note: This AI tool uses GPT technology for preliminary medication safety analysis. It should not replace professional medical advice.")
