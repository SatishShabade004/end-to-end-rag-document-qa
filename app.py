import streamlit as st
import os
import time
from dotenv import load_dotenv

# LangChain LLM
from langchain_groq import ChatGroq

# LangChain core
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Vector store + loaders + embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings


# ------------------------------------------------------------------
# ENV SETUP
# ------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()


# ------------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------------
st.set_page_config(page_title="Document Q&A (RAG)", layout="wide")
st.title("📄 Document Q&A using RAG (FAISS + Groq)")
st.caption("Ask questions strictly based on your PDF documents")


# ------------------------------------------------------------------
# LLM
# ------------------------------------------------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)


# ------------------------------------------------------------------
# PROMPT
# ------------------------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.
Answer the question strictly based on the provided context.
If the answer is not in the context, say "I don't know".

<context>
{context}
</context>

Question: {input}
"""
)


# ------------------------------------------------------------------
# VECTOR STORE (CACHED)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_vector_store():
    data_path = "./us_census"

    if not os.path.exists(data_path):
        st.error(f"❌ Folder not found: {data_path}")
        st.stop()

    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()

    if not documents:
        st.error("❌ No PDF files found in the folder")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ------------------------------------------------------------------
# CREATE / LOAD EMBEDDINGS
# ------------------------------------------------------------------
if st.button("📚 Create / Load Document Embeddings"):
    with st.spinner("Creating embeddings..."):
        st.session_state.vectors = load_vector_store()
    st.success("✅ Vector store ready!")


# ------------------------------------------------------------------
# QUESTION INPUT
# ------------------------------------------------------------------
question = st.text_input("🔍 Ask a question from your documents")


# ------------------------------------------------------------------
# RAG PIPELINE
# ------------------------------------------------------------------
if question:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please create document embeddings first.")
        st.stop()

    retriever = st.session_state.vectors.as_retriever(
        search_kwargs={"k": 4}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    start_time = time.process_time()
    answer = rag_chain.invoke(question)
    elapsed_time = time.process_time() - start_time

    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------
    st.subheader("🧠 Answer")
    st.write(answer)
    st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")

    # ------------------------------------------------------------------
    # SOURCE DOCUMENTS (FIXED)
    # ------------------------------------------------------------------
    with st.expander("📄 Retrieved Document Chunks"):
        docs = retriever.invoke(question)   # ✅ NEW LANGCHAIN API
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**Chunk {i}**")
            st.write(doc.page_content)
            st.divider()
