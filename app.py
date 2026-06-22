import os
import streamlit as st
from dotenv import load_dotenv

from loaders import load_documents
from rag_pipeline import build_vector_store, get_answer

load_dotenv()

# Load OpenAI API key from Streamlit Secrets

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(
page_title="Multi Document RAG Q&A Bot",
page_icon="📚",
layout="wide"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("📚 Multi Document RAG Q&A Bot")

uploaded_files = st.file_uploader(
"Upload PDF Documents",
type=["pdf"],
accept_multiple_files=True
)

if uploaded_files:

    with st.spinner("Processing documents..."):
        docs = load_documents(uploaded_files)
        vector_store = build_vector_store(docs)

    question = st.text_input("Ask a Question")

    if question:

        with st.spinner("Generating Answer..."):
            answer, sources = get_answer(
                vector_store,
                question
            )

        st.session_state.chat_history.append(
            (question, answer)
        )

        st.subheader("Answer")
        st.write(answer)

        if sources:
            st.subheader("Sources")

            for source in sources:
                st.write(f"• {source}")

if st.session_state.chat_history:
    st.subheader("Chat History")

    for question, answer in reversed(
        st.session_state.chat_history
    ):

        with st.expander(question):
            st.write(answer)
