from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
import pickle
import hashlib

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


def load_and_chunk_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def save_user_data(user_dir, name, obj):
    if name == "vectorstore":
        obj.save_local(os.path.join(user_dir, "vectorstore"))
    else:
        with open(os.path.join(user_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(obj, f)


def load_user_data(user_dir, name):
    if name == "vectorstore":
        path = os.path.join(user_dir, "vectorstore")
        if os.path.exists(path):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            return FAISS.load_local(
                path, embeddings, allow_dangerous_deserialization=True
            )
    else:
        path = os.path.join(user_dir, f"{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return None


# Simple username-password store (hashed for demonstration)
USERS = {
    "alice": hashlib.sha256("password123".encode()).hexdigest(),
    "bob": hashlib.sha256("secure456".encode()).hexdigest(),
}


def authenticate_user(username, password):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    return USERS.get(username) == hashed
