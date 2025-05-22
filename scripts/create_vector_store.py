# from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma

# # Load documents
# loader = DirectoryLoader(
#     path="data",
#     glob="**/*.txt",
#     loader_cls=TextLoader,
#     loader_kwargs={"encoding": "utf-8"}
# )
# documents = loader.load()

# # Add metadata
# for doc in documents:
#     if 'cv.txt' in doc.metadata.get('source', ''):
#         doc.metadata['query_type'] = 'personal,education,qualifications,skills'
#     elif 'images.txt' in doc.metadata.get('source', ''):
#         doc.metadata['query_type'] = 'image'

# # Split documents
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=300,
#     chunk_overlap=100,
#     length_function=len
# )
# chunks = text_splitter.split_documents(documents)

# # Create embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

# # Create vector store
# vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="chroma_db")
# vectorstore.persist()
# print("Vector store created successfully.")




# import os
# from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma

# # Load documents
# loader = DirectoryLoader(
#     path="data",
#     glob="**/*.txt",
#     loader_cls=TextLoader,
#     loader_kwargs={"encoding": "utf-8"}
# )
# documents = loader.load()

# # Add metadata
# for doc in documents:
#     if 'cv.txt' in doc.metadata.get('source', ''):
#         doc.metadata['query_type'] = 'personal,education,qualifications,skills'
#     elif 'images.txt' in doc.metadata.get('source', ''):
#         doc.metadata['query_type'] = 'image'

# # Split documents
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=200,
#     chunk_overlap=50,
#     length_function=len
# )
# chunks = text_splitter.split_documents(documents)

# # Create embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

# # Create vector store
# chroma_db_path = os.getenv("CHROMA_DB_PATH", "/chroma_db")
# vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=chroma_db_path)
# vectorstore.persist()
# print("Vector store created successfully.")


import logging
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def create_vector_store():
    logger.info("Starting vector store creation...")
    try:
        data_path = os.getenv("DATA_PATH", "./data")
        if not os.path.exists(data_path):
            logger.error(f"Data path {data_path} does not exist")
            raise FileNotFoundError(f"Data path {data_path} does not exist")

        logger.info(f"Loading documents from {data_path}...")
        loader = DirectoryLoader(
            data_path,
            glob="**/*.txt",
            loader_cls=lambda path: TextLoader(path, encoding='utf-8')
        )
        documents = []
        try:
            documents = loader.load()
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}", exc_info=True)
            raise

        if not documents:
            logger.error("No documents loaded")
            raise ValueError("No documents loaded")

        logger.info(f"Loaded {len(documents)} documents")

        logger.info("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        logger.info(f"Created {len(splits)} document splits")

        logger.info("Creating embeddings...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                cache_folder="/opt/render/.cache/huggingface"
            )
            logger.info("Embeddings loaded")
        except Exception as e:
            logger.error(f"Failed to load embeddings: {str(e)}", exc_info=True)
            raise

        # Use a local directory for Chroma DB if not on Render
        default_chroma_db_path = "./chroma_db"
        chroma_db_path = os.getenv("CHROMA_DB_PATH", default_chroma_db_path)
        try:
            # Ensure the directory is writable
            os.makedirs(os.path.dirname(chroma_db_path) or ".", exist_ok=True)
            with open(os.path.join(chroma_db_path, "test_write.txt"), "w") as f:
                f.write("test")
            os.remove(os.path.join(chroma_db_path, "test_write.txt"))
        except (OSError, PermissionError) as e:
            logger.warning(f"Cannot write to {chroma_db_path}: {str(e)}. Falling back to {default_chroma_db_path}")
            chroma_db_path = default_chroma_db_path
            os.makedirs(chroma_db_path, exist_ok=True)

        logger.info(f"Creating Chroma vector store at {chroma_db_path}...")
        try:
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=chroma_db_path
            )
            logger.info("Vector store created and persisted")
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}", exc_info=True)
            raise

    except Exception as e:
        logger.error(f"Error in create_vector_store: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    force_recreate = os.getenv("FORCE_RECREATE_DB", "false").lower() == "true"
    default_chroma_db_path = "./chroma_db"
    chroma_db_path = os.getenv("CHROMA_DB_PATH", default_chroma_db_path)
    if force_recreate or not os.path.exists(chroma_db_path):
        logger.info("Creating new vector store...")
        create_vector_store()
    else:
        logger.info(f"Vector store already exists at {chroma_db_path}, skipping creation")