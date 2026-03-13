from langchain_community.document_loaders import PyPDFLoader
import os

def load_documents(uploaded_files):

    docs = []

    os.makedirs("data", exist_ok=True)

    for file in uploaded_files:

        path = os.path.join("data", file.name)

        with open(path, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(path)

        documents = loader.load()

        docs.extend(documents)

    return docs