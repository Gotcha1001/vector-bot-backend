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




import os
import logging
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_vector_store():
    logger.debug(f"Resolved Chroma DB path: {os.path.abspath(chroma_db_path)}")
    # Check if Chroma DB already exists
    force_recreate = os.getenv("FORCE_RECREATE_DB", "false").lower() == "true"
    if os.path.exists(os.path.join(chroma_db_path, "chroma.sqlite3")) and not force_recreate:
        logger.info(f"Chroma DB already exists at {os.path.abspath(chroma_db_path)}. Skipping recreation.")
        return
    # Verify data directory exists
    data_path = "../data"
    logger.debug(f"Checking data directory: {os.path.abspath(data_path)}")
    if not os.path.exists(data_path):
        logger.error(f"Data directory not found: {os.path.abspath(data_path)}")
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    # Load documents
    logger.debug("Loading documents from data folder...")
    try:
        loader = DirectoryLoader(
            path="../data",
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents = loader.load()
        logger.debug(f"Loaded {len(documents)} documents")
        if not documents:
            logger.error("No documents loaded from data folder")
            raise ValueError("No documents loaded from data folder")
    except Exception as e:
        logger.error(f"Failed to load documents: {str(e)}", exc_info=True)
        raise
    # Add metadata
    for doc in documents:
        source = doc.metadata.get('source', '')
        if 'cv.txt' in source:
            doc.metadata['query_type'] = 'personal,education,qualifications,skills'
        elif 'images.txt' in source:
            doc.metadata['query_type'] = 'image'
    # Split documents
    logger.debug("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    logger.debug(f"Split {len(chunks)} chunks")
    # Create embeddings
    logger.debug("Creating embeddings...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
        logger.debug("Embeddings loaded")
    except Exception as e:
        logger.error(f"Failed to load embeddings: {str(e)}", exc_info=True)
        raise
    # Create vector store
    logger.debug(f"Creating vector store at {os.path.abspath(chroma_db_path)}...")
    try:
        vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=chroma_db_path)
        # No need for persist() as it's deprecated and auto-handled
        logger.info(f"Vector store created successfully at {os.path.abspath(chroma_db_path)}")
    except Exception as e:
        logger.error(f"Failed to create vector store: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Define chroma_db_path in global scope for the function
    chroma_db_path = os.getenv("CHROMA_DB_PATH", "../chroma_db")
    create_vector_store()