from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from config import (
CHUNK_SIZE,
CHUNK_OVERLAP,
EMBEDDING_MODEL,
LLM_MODEL
)

def build_vector_store(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store

def get_answer(vector_store, question):


    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    prompt = f"""
    
    You are an AI assistant answering questions strictly from the provided document context.
    
    If the answer is not present in the context, say:
    "I could not find that information in the uploaded documents."
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """


    response = llm.invoke(prompt)

    sources = []

    for doc in docs:
        source = doc.metadata.get("source")

        if source and source not in sources:
            sources.append(source)

    return response.content, sources
