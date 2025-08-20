import streamlit as st
import yaml
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# === Load OpenAI API key from Streamlit secrets ===
api_key = st.secrets["openai"]["api_key"]

# === Load config.yaml ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# === Synonyms dictionary ===
synonyms = {
    "education": ["education", "university", "universities", "degree", "degrees", "educations",
                  "education background", "educational background", "academic background", "qualifications"],
    "certifications": ["certification", "certifications", "certificate", "certificates", "exam", "exams"],
    "skills": ["skill", "skills", "expertise", "competence", "competences", "capability", "capabilities",
               "proficiencies", "programming languages", "programming", "soft skills", "strengths", "qualities"],
    "work_experience": ["projects", "project", "portfolio", "work experience", "work projects",
                        "career projects", "experience", "employment", "work", "job", "jobs",
                        "professional experience", "career experience", "work history"],
    "awards": ["award", "awards", "recognition", "recognitions", "honor", "honors"],
    "current_position": ["current position", "current job", "current role", "current employment", "current work"],
    "publications": ["publication", "publications", "paper", "papers", "article", "articles"],
    "languages": ["language", "languages", "spoken language", "spoken languages", "spoken"],
    "interests": ["interest", "interests", "hobby", "hobbies", "passions", "personal interests"],
    "references": ["reference", "references", "referee", "referees"],
    "contact": ["contact", "contact information", "contact info", "contact details"],
    "summary": ["summary", "profile", "introduction", "about me", "bio", "biography",
                "self-introduction", "self introduction", "self summary", "introduce yourself"],
    "achievements": ["achievement", "achievements", "award", "awards", "recognition", "recognitions"]
}

# === Function to map query synonyms ===
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

    for key, value in data.items():
        if key != "Example questions and answers":
            chunk_text = f"{key}:\n{yaml.dump(value, allow_unicode=True)}"
            documents.append(Document(page_content=chunk_text))

    if "Example questions and answers" in data:
        qas = data["Example questions and answers"]
        for i, qa in enumerate(qas):
            chunk_text = f"Example question {i+1}:\nq: {qa['q']}\na: {qa['a']}\n"
            documents.append(Document(page_content=chunk_text))

    return documents

# Correct path: pages/ -> sibling knowledge_base/
_chunks = load_yaml_as_documents("knowledge_base/profile.yaml")
st.write(f"📘 Loaded {len(_chunks)} chunks from YAML profile.")

# === Create vectorstore ===
@st.cache_resource
def create_vectorstore(_chunks):
    embeddings = HuggingFaceEmbeddings(model_name=config["embedding"]["model_name"])
    vectordb = Chroma.from_documents(_chunks, embeddings)
    return vectordb

vectorstore = create_vectorstore(_chunks)

# === Build RetrievalQA chain with ChatGPT ===
@st.cache_resource
def get_qa_chain():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=api_key
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": config["retriever"]["k"]})
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
    st.write(f"**Answer:** {result['result']}")
    st.session_state.history.append((query, result['result']))

    for q, a in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")
