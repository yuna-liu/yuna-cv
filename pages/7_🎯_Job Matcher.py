import streamlit as st
import yaml
from openai import OpenAI

# === OpenAI API key ===
api_key = st.secrets["openai"]["api_key"]
client = OpenAI(api_key=api_key)

# === Load YAML profile ===
@st.cache_data
def load_yaml(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Cannot find YAML file at: {path}")
        return {}

profile_data = load_yaml("knowledge_base/profile.yaml")
st.write(f"📘 Loaded CV profile with {len(profile_data)} sections.")

# === Convert YAML to text ===
def yaml_to_text(data):
    return yaml.dump(data, allow_unicode=True)

profile_text = yaml_to_text(profile_data)

# === System message for CV Q&A ===
cv_system_prompt = f"""
You are a helpful assistant who answers questions based on the following CV data.
If you cannot find an exact answer, say so and suggest related information.

CV Data:
{profile_text}
"""

st.title("🎯 Job Matcher")


# --- CV vs Job Match Section ---
st.header("📄 Compare CV to Job Description")

job_description = st.text_area(
    "Paste the job description here to see how my CV matches the role:",
    placeholder="Job title, responsibilities, requirements..."
)

if job_description:
    with st.spinner("Analyzing CV match with job description..."):
        comparison_prompt = f"""
You are a career assistant.

CV Data:
{profile_text}

Job Description:
{job_description}

Please:
1. List skills, experiences, and qualifications from the CV that match the job requirements.
2. Suggest areas in the CV that are missing for the job.
3. Provide an overall match score (0-100%) with a short explanation.
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant comparing a CV to a job description."},
                {"role": "user", "content": comparison_prompt}
            ],
            temperature=0
        )

        comparison_answer = response.choices[0].message.content
        st.write("**CV vs Job Comparison:**")
        st.write(comparison_answer)