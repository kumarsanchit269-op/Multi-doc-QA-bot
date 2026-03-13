# Multi-Document RAG Q&A Bot

A Retrieval-Augmented Generation (RAG) application that allows users to upload multiple PDF documents and ask questions about their content. The system retrieves relevant information from the documents and generates answers using an LLM.

---

## Features

- Upload multiple PDF documents
- AI-powered question answering
- Semantic search using vector embeddings
- Document chunking and retrieval
- Source citations for answers
- Streamlit interactive interface

---

## Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI API
- Chroma Vector Database
- PyPDF

---

## Project Architecture

Documents → Text Chunking → Embeddings → Vector Database → Retriever → LLM → Answer

---

## Project Structure
multi-doc-rag-bot
│
├── app.py
├── rag_pipeline.py
├── loaders.py
├── config.py
├── requirements.txt
├── README.md
│
├── data/
└── vector_store/


---

## Installation
Clone the repository

git clone https://github.com/yourusername/multi-doc-rag-bot.git

cd multi-doc-rag-bot

## Create a virtual environment

python -m venv venv

## Activate environment

venv\Scripts\activate


## Install dependencies
pip install -r requirements.txt


---

## Environment Variables

# Create a `.env` file
OPENAI_API_KEY=your_api_key_here

---

## Run the Application


streamlit run app.py


Open browser:


http://localhost:8501


---

## How It Works

1. Upload one or more PDF documents
2. Documents are split into smaller text chunks
3. Embeddings are created for each chunk
4. Chunks are stored in a vector database
5. User asks a question
6. Retriever finds the most relevant chunks
7. LLM generates an answer using the retrieved context

---

## Example Use Cases

- Research assistant
- Document analysis
- Study material Q&A
- Knowledge base search

---

## Future Improvements

- Support for DOCX and TXT files
- Chat-style conversation interface
- Persistent vector database loading
- Web deployment

---

## Author

Sanchit Kumar  
BCA AI/ML Student  
Aspiring AI Engineer
