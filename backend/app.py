





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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://vector-bot-frontend.vercel.app"]}})
app.config["DEBUG"] = True
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")
app.logger.setLevel(logging.DEBUG)

# Load embeddings
app.logger.debug("Loading HuggingFace embeddings...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
except Exception as e:
    app.logger.error(f"Failed to load embeddings: {str(e)}", exc_info=True)
    raise

# Specify Chroma DB path for Render's persistent disk
chroma_db_path = os.getenv("CHROMA_DB_PATH", "/chroma_db")  # Use persistent disk path
app.logger.debug(f"Using Chroma DB path: {chroma_db_path}")
try:
    vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
except Exception as e:
    app.logger.error(f"Failed to initialize Chroma DB at {chroma_db_path}: {str(e)}", exc_info=True)
    raise

# Initialize LLM with timeout
app.logger.debug("Initializing Gemini LLM...")
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        max_retries=3,
        timeout=30  # 30-second timeout
    )
except Exception as e:
    app.logger.error(f"Failed to initialize LLM: {str(e)}", exc_info=True)
    raise

# Set up memory and retriever
app.logger.debug("Setting up memory and retriever...")
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # Reduced k for performance

# Define prompt template
prompt_template = """
You are a friendly personal assistant with detailed knowledge about Wesley based on the provided documents. Use the context and chat history to answer the user's question accurately, naturally, and helpfully. Always prioritize the most relevant document for the query type.

### Instructions:
1. **Personal Details**: Use 'cv.txt' for questions about Wesley’s personal details.
2. **Qualifications and Education**: Use 'cv.txt' for qualifications or education.
3. **Skills and Work Experience**: Use 'cv.txt' and 'work.txt' for skills or experience.
4. **Image Queries**: Select the most relevant image from 'images.txt'.
5. **Hobbies and Activities**: Use 'hobbies.txt', 'music.txt', or 'personal.txt'.
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
    start_time = time.time()
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

        app.logger.debug(f"Retrieving documents for question: {question}")
        docs = retriever.invoke(question)
        app.logger.info(f"Retrieved {len(docs)} documents in {time.time() - start_time:.2f}s")

        app.logger.debug("Invoking conversation chain...")
        result = conversation_chain.invoke({"question": question})
        answer = result.get('answer', 'Sorry, I couldn’t find an answer.')
        app.logger.info(f"Answer generated in {time.time() - start_time:.2f}s: {answer[:100]}...")

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
        app.logger.debug(f"Returning response in {time.time() - start_time:.2f}s: {response}")
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error in /chat after {time.time() - start_time:.2f}s: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/clear-session', methods=['GET'])
def clear_session():
    start_time = time.time()
    try:
        app.logger.debug("Clearing session")
        session.clear()
        app.logger.debug(f"Session cleared in {time.time() - start_time:.2f}s")
        return jsonify({'message': 'Session cleared'})
    except Exception as e:
        app.logger.error(f"Error in /clear-session after {time.time() - start_time:.2f}s: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    app.logger.debug("Root endpoint called")
    return jsonify({'message': 'Welcome to Vector Bot Backend. Use /chat for queries.'}), 200

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)