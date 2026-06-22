from langchain_community.document_loaders import PyPDFLoader
import os

def load_documents(uploaded_files):
    docs = []

    os.makedirs("data", exist_ok=True)

    for uploaded_file in uploaded_files:

        file_path = os.path.join(
            "data",
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(file_path)

        documents = loader.load()

        for doc in documents:
            doc.metadata["source"] = uploaded_file.name

        docs.extend(documents)

    return docs
