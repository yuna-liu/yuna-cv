# app.py
import streamlit as st
import yaml
import os
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# --- Load YAML profile ---
yaml_path = "knowledge-base/profile.yaml"
if os.path.exists(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        profile_data = yaml.safe_load(f)
    profile_text = yaml.dump(profile_data, allow_unicode=True)
else:
    profile_text = ""
    st.warning("No profile.yaml found in knowledge-base folder.")

# System message with profile
system_message = f"""
You are a helpful assistant for Yuna.
Here is Yuna's profile:
{profile_text}
When relevant, use this information to answer personal questions.
"""

# --- Load PDFs into a vector database ---
pdf_folder = "knowledge-base"
docs = []
for file in os.listdir(pdf_folder):
    if file.lower().endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, file))
        docs.extend(loader.load())

if docs:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(api_key=st.secrets["openai"]["api_key"])
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=".chroma_db")
else:
    vectorstore = None
    st.warning("No PDFs found in knowledge-base folder.")

# --- OpenAI client ---
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# --- Streamlit UI ---
st.title("📚 Yuna's Personal & Document Assistant")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": system_message}
    ]

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # If vectorstore exists, search it
    retrieved_context = ""
    if vectorstore:
        results = vectorstore.similarity_search(prompt, k=3)
        retrieved_context = "\n\n".join([doc.page_content for doc in results])

    # Combine context
    combined_prompt = prompt
    if retrieved_context:
        combined_prompt += f"\n\nContext from documents:\n{retrieved_context}"

    # Get model response
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state["messages"] + [{"role": "user", "content": combined_prompt}]
    )

    reply = response.choices[0].message["content"]

    st.session_state["messages"].append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
