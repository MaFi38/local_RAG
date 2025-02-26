from langchain_community.document_loaders import  (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    TextLoader,
    UnstructuredCSVLoader,
    UnstructuredExcelLoader

)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain_ollama import OllamaEmbeddings
# import ollama


from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import os


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")

print("OPENAI_API_KEY:",OPENAI_API_KEY)

# EMBEDDING_MODEL = "nomic-embed-text"


# Initialize text splitter and embedding function
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
embedding_function = OpenAIEmbeddings()
# embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Initialize Chroma vector store
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)


def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='UTF-8')
    elif file_path.endswith('.csv'):
        loader = UnstructuredCSVLoader(file_path,mode="elements")
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        loader = UnstructuredExcelLoader(file_path,mode="elements")
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    documents = loader.load()
    return text_splitter.split_documents(documents)



def index_document_to_chroma(file_path: str, file_id: int,user_id:int) -> bool:
    try:
        # ollama.pull(EMBEDDING_MODEL)
        splits = load_and_split_document(file_path)

        # Add metadata to each split
        for split in splits:
            split.metadata['file_id'] = file_id
            split.metadata['user_id'] = user_id

        vectorstore.add_documents(splits)

        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False


def delete_doc_from_chroma(file_id: int,user_id:int):
    try:
        docs = vectorstore.get(where={"file_id": file_id} and {"user_id": user_id})
        print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")

        vectorstore._collection.delete(where={"file_id": file_id} and {"user_id": user_id})
        print(f"Deleted all documents with file_id {file_id}")

        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False
