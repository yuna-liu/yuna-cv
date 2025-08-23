import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import pandas as pd
import re
import json

# === OpenAI API key ===
api_key = st.secrets["openai"]["api_key"]
client = OpenAI(api_key=api_key)

st.title("🎯 Multi-CV Job Matcher (Test Version)")

st.markdown("""
Upload up to **5 PDF CVs** and paste the **Job Description** text.  
The AI will analyze each CV against the job description and provide:
- Matched skills/experience
- Missing skills/experience
- Overall match score
""")

# --- Upload multiple CVs ---
uploaded_files = st.file_uploader(
    "Upload PDF CVs (max 5)", type=["pdf"], accept_multiple_files=True
)

# --- Job description ---
job_description = st.text_area(
    "Paste the Job Description here:",
    placeholder="Job title, responsibilities, requirements..."
)

if uploaded_files and job_description:
    if len(uploaded_files) > 5:
        st.warning("Please upload up to 5 CVs only.")
    else:
        results = []

        for pdf_file in uploaded_files:
            # Read PDF text
            reader = PdfReader(pdf_file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

            # Create prompt for OpenAI
            prompt = f"""
You are a career assistant.

CV Text:
{text}

Job Description:
{job_description}

Please provide:
1. List of skills, experiences, and qualifications from the CV that match the job requirements.
2. List of skills/experience that are missing.
3. An overall match score (0-100%) with a short explanation.
Return the result in JSON format with keys: "matched", "missing", "score", "explanation".
"""

            # --- Call OpenAI ---
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful career assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            answer_text = response.choices[0].message.content

            # --- Try to parse JSON ---
            try:
                answer_json = json.loads(answer_text)
            except:
                answer_json = {
                    "matched": answer_text,
                    "missing": "",
                    "score": 0,
                    "explanation": ""
                }

            # --- Append to results ---
            results.append({
                "CV Filename": pdf_file.name,
                "Matched Skills": ", ".join(answer_json.get("matched")) if isinstance(answer_json.get("matched"), list) else str(answer_json.get("matched")),
                "Missing Skills": ", ".join(answer_json.get("missing")) if isinstance(answer_json.get("missing"), list) else str(answer_json.get("missing")),
                "Match Score": str(answer_json.get("score")),  # keep as string first
                "Explanation": str(answer_json.get("explanation"))
            })

        # --- After loop: display results ---
        df = pd.DataFrame(results)

        # Clean Match Score (keep only digits & dot)
        df["Match Score"] = df["Match Score"].apply(lambda x: re.sub(r"[^\d.]", "", str(x)))
        df["Match Score"] = pd.to_numeric(df["Match Score"], errors="coerce").fillna(0)

        # Sort descending
        df = df.sort_values("Match Score", ascending=False)

        st.subheader("CV Matching Results")
        st.dataframe(df)

        # Downloadable CSV report
        csv = df.to_csv(index=False)
        st.download_button("📥 Download Report", csv, file_name="cv_match_report.csv", mime="text/csv")
