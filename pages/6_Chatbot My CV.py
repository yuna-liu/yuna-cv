import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import yaml
from dotenv import load_dotenv
import streamlit as st
from langchain_core.documents import Document

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# === Load .env and config.yaml ===
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Synonyms dictionary (your existing one)
synonyms = {
    "education": ["education", "university", "universities", "degree", "degrees", "educations", "education background", "educational background", "academic background", "qualifications"],
    "certifications": ["certification", "certifications", "certificate", "certificates", "exam", "exams"],
    "skills": ["skill", "skills", "expertise", "competence", "competences", "capability", "capabilities", "proficiencies", "programming languages", "programming", "soft skills", "strengths", "qualities"],
    "work_experience:": ["projects", "project", "portfolio", "work experience", "work projects", "career projects", "work experience", "experience", "employment",  "work", "job", "jobs", "professional experience", "career experience", "work history"],
    "awards": ["award", "awards", "recognition", "recognitions", "honor", "honors"],
    "current_position": ["current position", "current job", "current role", "current employment", "current work"],
    "publications": ["publication", "publications", "paper", "papers", "article", "articles"],
    "languages": ["language", "languages", "spoken language", "spoken languages", "spoken"],
    "interests": ["interest", "interests", "hobby", "hobbies", "passions", "personal interests"],
    "references": ["reference", "references", "referee", "referees"],
    "contact": ["contact", "contact information", "contact info", "contact details"],
    "summary": ["summary", "profile", "introduction", "about me", "bio", "biography", "self-introduction", "self introduction", "self summary", "introduce yourself"],
    "achievements": ["achievement", "achievements", "award", "awards", "recognition", "recognitions"]
}

def map_query_synonyms(query: str, synonyms_dict: dict) -> str:
    query_lower = query.lower()
    for canonical, syn_list in synonyms_dict.items():
        for syn in syn_list:
            if syn in query_lower:
                query_lower = query_lower.replace(syn, canonical)
    return query_lower


# === Load YAML profile and convert to human-readable document chunks ===
@st.cache_resource
def load_yaml_as_documents(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    documents = []

    def to_str(obj):
        if isinstance(obj, dict):
            return "\n".join([f"{k}: {to_str(v)}" for k, v in obj.items()])
        elif isinstance(obj, list):
            return "\n- " + "\n- ".join([to_str(item) for item in obj])
        else:
            return str(obj)

    for section, content in data.items():
        if section in ["synonyms"]:  # Skip synonyms section
            continue
        doc_text = f"{section}:\n{to_str(content)}"
        documents.append(Document(page_content=doc_text, metadata={"section": section}))

    splitter = CharacterTextSplitter(
        chunk_size=config["document_loader"]["chunk_size"],
        chunk_overlap=config["document_loader"]["chunk_overlap"],
    )

    split_docs = []
    for doc in documents:
        split_docs.extend(splitter.split_documents([doc]))

    return split_docs



_chunks = load_yaml_as_documents("knowledge_base/profile.yaml")
st.write(f"📘 Loaded {len(_chunks)} chunks from YAML profile.")


# === Create vectorstore ===
@st.cache_resource
def create_vectorstore(_chunks):
    embeddings = HuggingFaceEmbeddings(model_name=config["embedding"]["model_name"])
    vectordb = Chroma.from_documents(_chunks, embeddings)
    return vectordb


vectorstore = create_vectorstore(_chunks)


# === Load local model and build Retrieval QA chain ===
@st.cache_resource
def get_qa_chain():
    pipe = pipeline(
        "text2text-generation",
        model=config["model"]["name"],
        tokenizer=config["model"]["name"],
        max_new_tokens=config["model"]["max_new_tokens"],
        temperature=config["model"]["temperature"],
        top_p=config["model"]["top_p"],
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    retriever = vectorstore.as_retriever(search_kwargs={"k": config["retriever"]["k"]})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


qa_chain = get_qa_chain()


# === Streamlit UI ===
st.title("🧠 CV Chatbot (YAML-based, Local FLAN-T5)")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask me anything about my education, certifications, skills, or work experience:")

if query:
    normalized_query = map_query_synonyms(query, synonyms)
    with st.spinner("Generating answer..."):
        result = qa_chain.invoke({"query": normalized_query})
    st.write(f"**Answer:** {result['result']}")
    st.session_state.history.append((query, result['result']))

    for q, a in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:**\n{a}")
        st.markdown("---")
