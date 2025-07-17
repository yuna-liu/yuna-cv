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
import pandas as pd
from langchain_core.documents import Document


from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
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

# === Step 1: Load and split documents ===
@st.cache_resource
def load_and_split_docs():
    loader = DirectoryLoader(
        config["document_loader"]["folder"],
        glob=config["document_loader"]["pattern"],
        loader_cls=PyPDFLoader,
    )
    docs = loader.load()
    splitter = CharacterTextSplitter(
        chunk_size=config["document_loader"]["chunk_size"],
        chunk_overlap=config["document_loader"]["chunk_overlap"],
    )
    return splitter.split_documents(docs)


@st.cache_resource
def load_csv_as_documents(path: str, source: str) -> list:
    df = pd.read_csv(path)
    docs = []

    for _, row in df.iterrows():
        text = f"Source: {source} | " + " | ".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(Document(page_content=text))

    return docs


pdf_chunks = load_and_split_docs()
cert_chunks = load_csv_as_documents("data/certifications.csv", source="Certifications")
edu_chunks = load_csv_as_documents("data/education.csv", source="Education")

_chunks = pdf_chunks + cert_chunks + edu_chunks

st.write(f"📚 Loaded {len(_chunks)} document chunks (PDF + CSVs)")



# === Step 2: Create vectorstore ===
@st.cache_resource
def create_vectorstore(_chunks):
    embeddings = HuggingFaceEmbeddings(model_name=config["embedding"]["model_name"])
    vectordb = Chroma.from_documents(_chunks, embeddings)
    return vectordb

vectorstore = create_vectorstore(_chunks)

# === Step 3: Load local model and build Retrieval QA chain ===
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

# === Step 4: Streamlit UI ===
st.title("📄 My CV Chatbot (Local FLAN-T5)")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask me anything about my CV:")

if query:
    with st.spinner("Generating answer..."):
        result = qa_chain.invoke({"query": query})
    st.session_state.history.append((query, result["result"]))
    st.write(f"**Answer:** {result['result']}")

    for q, a in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")
