





# import logging
# import os
# from flask import Flask, request, jsonify, session
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# # Set up logging
# logging.basicConfig(level=logging.DEBUG)

# # Load environment variables
# load_dotenv()

# # Initialize Flask app
# app = Flask(__name__)
# app.config["DEBUG"] = True
# app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")
# app.logger.setLevel(logging.DEBUG)

# # Load embeddings
# app.logger.debug("Loading HuggingFace embeddings...")
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

# # Specify the absolute path to the chroma_db directory in the project root
# root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# chroma_db_path = os.path.join(root_dir, "chroma_db")
# app.logger.debug(f"Using Chroma DB path: {chroma_db_path}")

# # Load Chroma vector store
# app.logger.debug("Loading Chroma vector store...")
# vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)

# # Initialize LLM
# app.logger.debug("Initializing Gemini LLM...")
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     temperature=0.7,
#     google_api_key=os.getenv("GOOGLE_API_KEY"),
#     max_retries=3
# )

# # Set up memory and retriever
# app.logger.debug("Setting up memory and retriever...")
# memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# # Define prompt template
# prompt_template = """
# You are a friendly personal assistant with detailed knowledge about Wesley based on the provided documents. Use the context and chat history to answer the user's question accurately, naturally, and helpfully. Always prioritize the most relevant document for the query type.

# ### Instructions:
# 1. **Personal Details**:
#    - For questions about Wesley’s personal details, use 'cv.txt'.
#    - Example: If asked, "When was he born?" respond, "Wesley was born on 22 January 1982, as noted in his CV."

# 2. **Qualifications and Education**:
#    - For qualifications or education, use 'cv.txt'.
#    - Example: If asked, "What did he study?" respond, "Wesley studied at the University of KwaZulu-Natal, earning a BPMus Jazz and Popular Music Honours in 2008."

# 3. **Skills and Work Experience**:
#    - For skills or experience, use 'cv.txt' and 'work.txt'.
#    - Example: If asked, "What are his technical skills?" respond, "Wesley is skilled in JavaScript, TypeScript, React, Python, and more."

# 4. **Image Queries**:
#    - For image requests, select the most relevant image from 'images.txt'.
#    - Example: For "Wesley as an elf", return "Wesley coding intensely" (elf.jpg).

# 5. **Hobbies and Activities**:
#    - For hobbies, use 'hobbies.txt', 'music.txt', or 'personal.txt'.
#    - Example: If asked, "What does he do in his spare time?" respond, "Wesley loves playing piano and hiking."

# 6. **Tone and Style**:
#    - Speak fondly of Wesley, keep responses concise (2–3 sentences), and avoid speculation.

# ### Chat History:
# {chat_history}

# ### Context:
# {context}

# ### Question:
# {question}

# ### Answer:
# """
# prompt = PromptTemplate(
#     template=prompt_template,
#     input_variables=["chat_history", "context", "question"]
# )

# # Create conversational chain
# app.logger.debug("Creating conversational chain...")
# conversation_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=retriever,
#     memory=memory,
#     return_source_documents=False,
#     combine_docs_chain_kwargs={"prompt": prompt}
# )

# @app.route('/health')
# def health():
#     app.logger.debug("Health check endpoint called")
#     return jsonify({'status': 'ok'}), 200

# @app.route('/chat', methods=['POST'])
# def chat():
#     try:
#         app.logger.debug("Received /chat request")
#         question = request.json.get('question', '')
#         app.logger.debug(f"Question: {question}")
#         if not question:
#             app.logger.warning("No question provided")
#             return jsonify({'error': 'No question provided'}), 400

#         if 'used_images' not in session:
#             app.logger.debug("Initializing session.used_images")
#             session['used_images'] = []

#         app.logger.debug(f"Retrieving documents for question: {question}")
#         docs = retriever.get_relevant_documents(question)
#         app.logger.info(f"Retrieved {len(docs)} documents")

#         app.logger.debug("Invoking conversation chain...")
#         result = conversation_chain.invoke({"question": question})
#         answer = result.get('answer', 'Sorry, I couldn’t find an answer.')
#         app.logger.info(f"Chain result: answer={answer[:100]}...")

#         image_url = None
#         if "image" in question.lower() or "picture" in question.lower():
#             app.logger.debug("Processing image query")
#             for doc in docs:
#                 if 'images.txt' in doc.metadata.get('source', '') and "/images/" in doc.page_content:
#                     image_name = doc.page_content.split("/images/")[1].split()[0].strip(',.')
#                     description = doc.page_content.split(':')[0].lower().strip()
#                     keywords = description.split() + [image_name.lower().replace('.jpg', '').replace('.png', '')]
#                     if any(keyword in question.lower() for keyword in keywords):
#                         if image_name not in session['used_images']:
#                             image_url = f"/images/{image_name}"
#                             session['used_images'].append(image_name)
#                             app.logger.info(f"Selected image: {image_url}")
#                             break

#             if not image_url:
#                 for doc in docs:
#                     if 'images.txt' in doc.metadata.get('source', '') and "/images/" in doc.page_content:
#                         image_name = doc.page_content.split("/images/")[1].split()[0].strip(',.')
#                         if image_name not in session['used_images']:
#                             image_url = f"/images/{image_name}"
#                             session['used_images'].append(image_name)
#                             app.logger.info(f"Fallback image: {image_url}")
#                             break

#             if not image_url:
#                 app.logger.debug("Resetting used_images")
#                 session['used_images'] = []
#                 for doc in docs:
#                     if 'images.txt' in doc.metadata.get('source', '') and "/images/" in doc.page_content:
#                         image_name = doc.page_content.split("/images/")[1].split()[0].strip(',.')
#                         image_url = f"/images/{image_name}"
#                         session['used_images'].append(image_name)
#                         app.logger.info(f"Reset and selected image: {image_url}")
#                         break

#         response = {'answer': answer}
#         if image_url:
#             response['image_url'] = image_url
#         app.logger.debug(f"Returning response: {response}")
#         return jsonify(response)
#     except Exception as e:
#         app.logger.error(f"Error in /chat: {str(e)}", exc_info=True)
#         return jsonify({'error': str(e)}), 500

# @app.route('/clear-session', methods=['GET'])
# def clear_session():
#     app.logger.debug("Clearing session")
#     session.clear()
#     return jsonify({'message': 'Session cleared'})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)




import logging
import os
import time
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import psutil

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://vector-bot-frontend.vercel.app"]}})
app.config["DEBUG"] = True
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = '/tmp/flask_session'
if not os.path.exists('/tmp/flask_session'):
    os.makedirs('/tmp/flask_session')
    app.logger.debug("Created /tmp/flask_session directory")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")
Session(app)
app.logger.setLevel(logging.DEBUG)

# Log memory usage
app.logger.info(f"Memory usage before initialization: {psutil.virtual_memory().percent}%")

# Load embeddings
app.logger.debug("Loading HuggingFace embeddings...")
start_time = time.time()
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_kwargs={"device": "cpu"}
    )
    app.logger.debug(f"Embeddings loaded in {time.time() - start_time:.2f} seconds")
except Exception as e:
    app.logger.error(f"Failed to load embeddings: {str(e)}", exc_info=True)
    raise
app.logger.info(f"Memory usage after embeddings: {psutil.virtual_memory().percent}%")

# Specify Chroma DB path using Render's project root
project_root = "/opt/render/project/src"
chroma_db_path = os.getenv("CHROMA_DB_PATH", os.path.join(project_root, "chroma_db"))
app.logger.debug(f"Resolved Chroma DB path: {os.path.abspath(chroma_db_path)}")

# Initialize Chroma DB
app.logger.debug("Initializing Chroma DB...")
start_time = time.time()
try:
    os.makedirs(chroma_db_path, exist_ok=True)  # Ensure directory exists
    os.environ["ANONYMIZED_TELEMETRY"] = "False"  # Disable telemetry
    vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
    app.logger.debug(f"Chroma DB initialized in {time.time() - start_time:.2f} seconds")
    if os.path.exists(os.path.join(chroma_db_path, "chroma.sqlite3")):
        app.logger.info("Chroma DB file found")
    else:
        app.logger.error("Chroma DB file not found")
        raise FileNotFoundError("Chroma DB file not found")
except Exception as e:
    app.logger.error(f"Failed to initialize Chroma DB at {chroma_db_path}: {str(e)}", exc_info=True)
    raise
app.logger.info(f"Memory usage after Chroma DB: {psutil.virtual_memory().percent}%")

# Initialize LLM with timeout
app.logger.debug("Initializing Gemini LLM...")
start_time = time.time()
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        max_retries=3,
        timeout=90
    )
    app.logger.debug(f"LLM initialized in {time.time() - start_time:.2f} seconds")
except Exception as e:
    app.logger.error(f"Failed to initialize LLM: {str(e)}", exc_info=True)
    raise

# Set up memory and retriever
app.logger.debug("Setting up memory and retriever...")
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Define prompt template
prompt_template = """
You are a friendly personal assistant with detailed knowledge about Wesley based on the provided documents. Use the context and chat history to answer the user's question accurately, naturally, and helpfully. Always prioritize the most relevant document for the query type.

### Instructions:
1. **Personal Details**: Use 'cv.txt' for questions about Wesley’s personal details.
2. **Qualifications and Education**: Use 'cv.txt' for qualifications or education.
3. **Skills and Work Experience**: Use 'cv.txt' and 'work.txt' for skills or experience.
4. **Image Queries**: Select the most relevant image from 'images.txt'.
5. **Hobbies and Activities**: Use 'hobbies.txt', 'music.txt', or 'certificates.txt'.
6. **Tone and Style**: Speak fondly of Wesley, keep responses concise (2–3 sentences), and avoid speculation.

### Chat History:
{chat_history}

### Context:
{context}

### Question:
{question}

### Answer:
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["chat_history", "context", "question"]
)

# Create conversational chain
app.logger.debug("Creating conversational chain...")
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=False,
    combine_docs_chain_kwargs={"prompt": prompt}
)

@app.route('/health')
def health():
    app.logger.debug("Health check endpoint called")
    return jsonify({'status': 'ok'}), 200

@app.route('/chat', methods=['POST'])
def chat():
    total_start = time.time()
    try:
        app.logger.debug("Received /chat request")
        question = request.json.get('question', '')
        app.logger.debug(f"Question: {question}")
        if not question:
            app.logger.warning("No question provided")
            return jsonify({'error': 'No question provided'}), 400

        try:
            if 'used_images' not in session:
                app.logger.debug("Initializing session.used_images")
                session['used_images'] = []
        except Exception as e:
            app.logger.error(f"Session initialization error: {str(e)}", exc_info=True)
            session.clear()
            session['used_images'] = []

        # Document retrieval with timing
        app.logger.debug(f"Retrieving documents for question: {question}")
        retrieval_start = time.time()
        docs = retriever.invoke(question)
        retrieval_time = time.time() - retrieval_start
        app.logger.info(f"Retrieved {len(docs)} documents in {retrieval_time:.2f} seconds")

        # Conversation chain invocation with timing
        app.logger.debug("Invoking conversation chain...")
        chain_start = time.time()
        result = conversation_chain.invoke({"question": question})
        chain_time = time.time() - chain_start
        app.logger.info(f"Conversation chain invocation took {chain_time:.2f} seconds")
        answer = result.get('answer', 'Sorry, I couldn’t find an answer.')
        app.logger.debug(f"Generated answer: {answer[:100]}...")

        image_url = None
        if "image" in question.lower() or "picture" in question.lower():
            app.logger.debug("Processing image query")
            for doc in docs:
                if 'images.txt' in doc.metadata.get('source', '') and "/images/" in doc.page_content:
                    image_name = doc.page_content.split("/images/")[1].split()[0].strip(',.')
                    description = doc.page_content.split(':')[0].lower().strip()
                    keywords = description.split() + [image_name.lower().replace('.jpg', '').replace('.png', '')]
                    if any(keyword in question.lower() for keyword in keywords):
                        if image_name not in session['used_images']:
                            image_url = f"https://vector-bot-frontend.vercel.app/images/{image_name}"
                            session['used_images'].append(image_name)
                            app.logger.info(f"Selected image: {image_url}")
                            break

            if not image_url:
                for doc in docs:
                    if 'images.txt' in doc.metadata.get('source', '') and "/images/" in doc.page_content:
                        image_name = doc.page_content.split("/images/")[1].split()[0].strip(',.')
                        if image_name not in session['used_images']:
                            image_url = f"https://vector-bot-frontend.vercel.app/images/{image_name}"
                            session['used_images'].append(image_name)
                            app.logger.info(f"Fallback image: {image_url}")
                            break

            if not image_url:
                app.logger.debug("Resetting used_images")
                session['used_images'] = []
                for doc in docs:
                    if 'images.txt' in doc.metadata.get('source', '') and "/images/" in doc.page_content:
                        image_name = doc.page_content.split("/images/")[1].split()[0].strip(',.')
                        image_url = f"https://vector-bot-frontend.vercel.app/images/{image_name}"
                        session['used_images'].append(image_name)
                        app.logger.info(f"Reset and selected image: {image_url}")
                        break

        response = {'answer': answer}
        if image_url:
            response['image_url'] = image_url

        total_time = time.time() - total_start
        app.logger.info(f"Total time for /chat request: {total_time:.2f} seconds")
        app.logger.debug(f"Returning response: {response}")
        return jsonify(response)
    except Exception as e:
        total_time = time.time() - total_start
        app.logger.error(f"Error in /chat after {total_time:.2f} seconds: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/clear-session', methods=['GET'])
def clear_session():
    start_time = time.time()
    try:
        app.logger.debug("Clearing session")
        session.clear()
        app.logger.debug(f"Session cleared in {time.time() - start_time:.2f} seconds")
        return jsonify({'message': 'Session cleared'})
    except Exception as e:
        app.logger.error(f"Error in /clear-session after {time.time() - start_time:.2f} seconds: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    app.logger.debug("Root endpoint called")
    return jsonify({'message': 'Welcome to Vector Bot Backend. Use /chat for queries.'}), 200

@app.route('/debug')
def debug():
    return jsonify({
        'cwd': os.getcwd(),
        'chroma_path': os.path.abspath(os.path.join(project_root, 'chroma_db')),
        'chroma_exists': os.path.exists(os.path.join(project_root, 'chroma_db')),
        'chroma_file_exists': os.path.exists(os.path.join(project_root, 'chroma_db', 'chroma.sqlite3')),
        'data_path': os.path.abspath(os.path.join(project_root, 'data')),
        'data_exists': os.path.exists(os.path.join(project_root, 'data')),
        'root_contents': os.listdir(project_root)
    })

@app.route('/data-files')
def data_files():
    data_path = os.path.join(project_root, "data")
    try:
        files = os.listdir(data_path)
        return jsonify({"data_files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/memory')
def memory():
    return jsonify({'memory_percent': psutil.virtual_memory().percent})

@app.route('/disk')
def disk():
    stat = os.statvfs('/')
    total = stat.f_frsize * stat.f_blocks
    free = stat.f_frsize * stat.f_bavail
    return jsonify({"total_bytes": total, "free_bytes": free})

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)