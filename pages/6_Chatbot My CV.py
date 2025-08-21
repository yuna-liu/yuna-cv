import streamlit as st
import yaml
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

# === OpenAI API key ===
api_key = st.secrets["openai"]["api_key"]

# === Synonyms dictionary ===
synonyms = {
    "education": ["education", "university", "degree", "academic background", "qualifications"],
    "certifications": ["certification", "certificate", "exam"],
    "skills": ["skills", "expertise", "competences", "programming languages", "soft skills"],
    "work_experience": ["work experience", "experience", "projects", "career projects", "job", "jobs"],
    "awards": ["awards", "recognition", "honor"],
    "current_position": ["current role", "current position", "current job"],
    "publications": ["publications", "papers", "articles"],
    "languages": ["languages", "spoken language"],
    "interests": ["interests", "hobbies", "passions"],
    "contact": ["contact", "contact information"],
    "summary": ["summary", "profile", "introduction", "bio"],
    "achievements": ["achievements", "award", "recognition"]
}

def map_query_synonyms(query: str, synonyms_dict: dict) -> str:
    query_lower = query.lower()
    for canonical, syn_list in synonyms_dict.items():
        for syn in syn_list:
            if syn in query_lower:
                query_lower = query_lower.replace(syn, canonical)
    return query_lower

# === Load YAML profile as documents ===
@st.cache_resource
def load_yaml_as_documents(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Cannot find YAML file at: {path}")
        return []

    documents = []

    # Top-level keys
    for key, value in data.items():
        if key != "Example questions and answers":
            chunk_text = f"{key}:\n{yaml.dump(value, allow_unicode=True)}"
            documents.append(Document(page_content=chunk_text, metadata={"key": key}))

    # Add example Q&A separately
    if "Example questions and answers" in data:
        for i, qa in enumerate(data["Example questions and answers"]):
            chunk_text = f"Q: {qa['q']}\nA: {qa['a']}"
            documents.append(Document(page_content=chunk_text, metadata={"key": "Example Q&A"}))

    return documents

_chunks = load_yaml_as_documents("knowledge_base/profile.yaml")
st.write(f"📘 Loaded {len(_chunks)} chunks from YAML profile.")

# === Create vectorstore ===
# === Create vectorstore ===
@st.cache_resource
def create_vectorstore(_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        # Force FAISS if on Streamlit Cloud or Chroma fails
        if os.environ.get("STREAMLIT_RUNTIME") == "cloud":
            st.info("Using FAISS (cloud mode)")
            vectordb = FAISS.from_documents(_chunks, embeddings)
        else:
            st.info("Using Chroma (local mode)")
            vectordb = Chroma.from_documents(_chunks, embeddings, persist_directory="chroma_db")
        return vectordb
    except RuntimeError as e:
        st.warning(f"Chroma failed: {e} — switching to FAISS.")
        return FAISS.from_documents(_chunks, embeddings)

vectorstore = create_vectorstore(_chunks)

# === Build RetrievalQA chain with ChatGPT ===
@st.cache_resource
def get_qa_chain():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=api_key
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

qa_chain = get_qa_chain()

# === Streamlit UI ===
st.title("🧠 CV Chatbot (ChatGPT-based)")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask me anything about my education, certifications, skills, or work experience:")

if query:
    normalized_query = map_query_synonyms(query, synonyms)
    with st.spinner("Generating answer..."):
        result = qa_chain({"query": normalized_query})

    # Fallback: if result is empty, check top-level keys
    if not result.get("result"):
        for doc in _chunks:
            if doc.metadata.get("key") == normalized_query:
                result["result"] = doc.page_content
                break

    st.write(f"**Answer:** {result['result']}")
    st.session_state.history.append((query, result['result']))

    for q, a in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")
