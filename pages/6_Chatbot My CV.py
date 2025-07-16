import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline # 👈 This is for local use

from transformers import pipeline  # 👈 Local model pipeline

# 1. Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 2. Load and split documents
@st.cache_resource
def load_and_split_docs():
    loader = DirectoryLoader(
        "knowledge-base",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

_chunks = load_and_split_docs()
st.write(f"Loaded {len(_chunks)} document chunks.")

# 3. Create vectorstore
@st.cache_resource
def create_vectorstore(_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(_chunks, embeddings, persist_directory="db")
    return vectordb

vectorstore = create_vectorstore(_chunks)

# 4. Use local FLAN-T5 model
@st.cache_resource
def get_qa_chain():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_new_tokens=256,
        temperature=0.5,
        top_p=0.95
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

qa_chain = get_qa_chain()

# 5. Streamlit interface
st.title("📄 My CV Chatbot (Local Flan-T5)")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask me anything about my CV:")

if query:
    with st.spinner("Generating answer..."):
        result = qa_chain.run(query)
    st.session_state.history.append((query, result))
    st.write(f"**Answer:** {result}")

    for q, a in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")
