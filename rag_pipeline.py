from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, LLM_MODEL

def build_vector_store(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vector_store = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="vector_store"
    )

    return vector_store

def get_answer(vector_store, question):

    retriever = vector_store.as_retriever()

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    prompt = f"""
    ```
    
    You are an AI assistant answering questions based only on the provided documents.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """

    response = llm.invoke(prompt)

    sources = []

    for doc in docs:
        if "source" in doc.metadata:
            sources.append(doc.metadata["source"])

    return response.content, list(set(sources))
