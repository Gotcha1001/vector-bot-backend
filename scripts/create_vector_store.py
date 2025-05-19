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
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Check if Chroma DB already exists
chroma_db_path = os.getenv("CHROMA_DB_PATH", "/chroma_db")
force_recreate = os.getenv("FORCE_RECREATE_DB", "false").lower() == "true"

if os.path.exists(os.path.join(chroma_db_path, "chroma.sqlite3")) and not force_recreate:
    print(f"Chroma DB already exists at {chroma_db_path}. Skipping recreation.")
else:
    # Load documents
    loader = DirectoryLoader(
        path="data",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()

    # Add metadata
    for doc in documents:
        if 'cv.txt' in doc.metadata.get('source', ''):
            doc.metadata['query_type'] = 'personal,education,qualifications,skills'
        elif 'images.txt' in doc.metadata.get('source', ''):
            doc.metadata['query_type'] = 'image'

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

    # Create vector store
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=chroma_db_path)
    vectorstore.persist()
    print(f"Vector store created successfully at {chroma_db_path}.")