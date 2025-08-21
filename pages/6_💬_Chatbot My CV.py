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

# === System message ===
system_prompt = f"""
You are a helpful assistant who answers questions based on the following CV data.
If you cannot find an exact answer, say so and suggest related information.

CV Data:
{profile_text}
"""

# === Streamlit UI ===
st.title("💬 CV Chatbot (OpenAI-based)")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask me anything about my CV:")

if query:
    with st.spinner("Generating answer..."):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0
        )

        answer = response.choices[0].message.content
        st.write(f"**Answer:** {answer}")
        st.session_state.history.append((query, answer))

# === Show history ===
if st.session_state.history:
    st.markdown("---")
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
