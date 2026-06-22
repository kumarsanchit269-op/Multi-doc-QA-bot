import streamlit as st
from dotenv import load_dotenv

from loaders import load_documents
from rag_pipeline import build_vector_store, get_answer

load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Multi Document RAG QnA Bot")

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    st.write("Processing documents...")

    docs = load_documents(uploaded_files)

    vector_store = build_vector_store(docs)

    question = st.text_input("Ask a question")

    if question:

        answer, sources = get_answer(
            vector_store,
            question
        )

        st.session_state.chat_history.append((question, answer))
        st.write("### Answer")
        st.write(answer)

        st.write("### Sources")
        if sources:
            st.write("### Sources")
        for s in sources:
            st.write(f"- {s}")
