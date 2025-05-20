





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

























# import logging
# import os
# import time
# from flask import Flask, request, jsonify, session
# from flask_cors import CORS
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
# CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://vector-bot-frontend.vercel.app"]}})
# app.config["DEBUG"] = True
# app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")
# app.logger.setLevel(logging.DEBUG)

# # Load embeddings
# app.logger.debug("Loading HuggingFace embeddings...")
# try:
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
# except Exception as e:
#     app.logger.error(f"Failed to load embeddings: {str(e)}", exc_info=True)
#     raise

# # Specify Chroma DB path using a relative path
# chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
# app.logger.debug(f"Using Chroma DB path: {chroma_db_path}")

# # Check if the Chroma DB directory exists
# if not os.path.exists(chroma_db_path):
#     app.logger.error(f"Chroma DB path {chroma_db_path} does not exist")
#     raise FileNotFoundError(f"Chroma DB path {chroma_db_path} does not exist")

# # Initialize Chroma DB with the existing directory
# try:
#     vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
# except Exception as e:
#     app.logger.error(f"Failed to initialize Chroma DB at {chroma_db_path}: {str(e)}", exc_info=True)
#     raise

# # Initialize LLM with timeout
# app.logger.debug("Initializing Gemini LLM...")
# try:
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-1.5-flash",
#         temperature=0.7,
#         google_api_key=os.getenv("GOOGLE_API_KEY"),
#         max_retries=3,
#         timeout=30  # 30-second timeout
#     )
# except Exception as e:
#     app.logger.error(f"Failed to initialize LLM: {str(e)}", exc_info=True)
#     raise

# # Set up memory and retriever
# app.logger.debug("Setting up memory and retriever...")
# memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # Reduced k for performance

# # Define prompt template
# prompt_template = """
# You are a friendly personal assistant with detailed knowledge about Wesley based on the provided documents. Use the context and chat history to answer the user's question accurately, naturally, and helpfully. Always prioritize the most relevant document for the query type.

# ### Instructions:
# 1. **Personal Details**: Use 'cv.txt' for questions about Wesley’s personal details.
# 2. **Qualifications and Education**: Use 'cv.txt' for qualifications or education.
# 3. **Skills and Work Experience**: Use 'cv.txt' and 'work.txt' for skills or experience.
# 4. **Image Queries**: Select the most relevant image from 'images.txt'.
# 5. **Hobbies and Activities**: Use 'hobbies.txt', 'music.txt', or 'personal.txt'.
# 6. **Tone and Style**: Speak fondly of Wesley, keep responses concise (2–3 sentences), and avoid speculation.

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
#     start_time = time.time()
#     try:
#         app.logger.debug("Received /chat request")
#         question = request.json.get('question', '')
#         app.logger.debug(f"Question: {question}")
#         if not question:
#             app.logger.warning("No question provided")
#             return jsonify({'error': 'No question provided'}), 400

#         try:
#             if 'used_images' not in session:
#                 app.logger.debug("Initializing session.used_images")
#                 session['used_images'] = []
#         except Exception as e:
#             app.logger.error(f"Session initialization error: {str(e)}", exc_info=True)
#             session.clear()
#             session['used_images'] = []

#         app.logger.debug(f"Retrieving documents for question: {question}")
#         docs = retriever.invoke(question)
#         app.logger.info(f"Retrieved {len(docs)} documents in {time.time() - start_time:.2f}s")

#         app.logger.debug("Invoking conversation chain...")
#         result = conversation_chain.invoke({"question": question})
#         answer = result.get('answer', 'Sorry, I couldn’t find an answer.')
#         app.logger.info(f"Answer generated in {time.time() - start_time:.2f}s: {answer[:100]}...")

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
#                             image_url = f"https://vector-bot-frontend.vercel.app/images/{image_name}"
#                             session['used_images'].append(image_name)
#                             app.logger.info(f"Selected image: {image_url}")
#                             break

#             if not image_url:
#                 for doc in docs:
#                     if 'images.txt' in doc.metadata.get('source', '') and "/images/" in doc.page_content:
#                         image_name = doc.page_content.split("/images/")[1].split()[0].strip(',.')
#                         if image_name not in session['used_images']:
#                             image_url = f"https://vector-bot-frontend.vercel.app/images/{image_name}"
#                             session['used_images'].append(image_name)
#                             app.logger.info(f"Fallback image: {image_url}")
#                             break

#             if not image_url:
#                 app.logger.debug("Resetting used_images")
#                 session['used_images'] = []
#                 for doc in docs:
#                     if 'images.txt' in doc.metadata.get('source', '') and "/images/" in doc.page_content:
#                         image_name = doc.page_content.split("/images/")[1].split()[0].strip(',.')
#                         image_url = f"https://vector-bot-frontend.vercel.app/images/{image_name}"
#                         session['used_images'].append(image_name)
#                         app.logger.info(f"Reset and selected image: {image_url}")
#                         break

#         response = {'answer': answer}
#         if image_url:
#             response['image_url'] = image_url
#         app.logger.debug(f"Returning response in {time.time() - start_time:.2f}s: {response}")
#         return jsonify(response)
#     except Exception as e:
#         app.logger.error(f"Error in /chat after {time.time() - start_time:.2f}s: {str(e)}", exc_info=True)
#         return jsonify({'error': str(e)}), 500

# @app.route('/clear-session', methods=['GET'])
# def clear_session():
#     start_time = time.time()
#     try:
#         app.logger.debug("Clearing session")
#         session.clear()
#         app.logger.debug(f"Session cleared in {time.time() - start_time:.2f}s")
#         return jsonify({'message': 'Session cleared'})
#     except Exception as e:
#         app.logger.error(f"Error in /clear-session after {time.time() - start_time:.2f}s: {str(e)}", exc_info=True)
#         return jsonify({'error': str(e)}), 500

# @app.route('/')
# def index():
#     app.logger.debug("Root endpoint called")
#     return jsonify({'message': 'Welcome to Vector Bot Backend. Use /chat for queries.'}), 200

# if __name__ == '__main__':
#     port = int(os.getenv("PORT", 5000))
#     app.run(host='0.0.0.0', port=port, debug=True)































# import logging
# import os
# import time
# from flask import Flask, request, jsonify, session
# from flask_cors import CORS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# # Set up logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
#     handlers=[
#         logging.StreamHandler()  # Ensure logs go to stdout for Render
#     ]
# )
# logger = logging.getLogger(__name__)

# # Load environment variables
# logger.info("Loading environment variables...")
# load_dotenv()

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://vector-bot-frontend.vercel.app"]}})
# app.config["DEBUG"] = True
# app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")
# if not app.secret_key:
#     logger.error("FLASK_SECRET_KEY is not set")
#     raise ValueError("FLASK_SECRET_KEY is not set")
# logger.info("Flask app initialized")

# try:
#     # Load embeddings
#     logger.debug("Loading HuggingFace embeddings...")
#     start_time = time.time()
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
#         model_kwargs={"device": "cpu"}
#     )
#     logger.debug(f"Embeddings loaded in {time.time() - start_time:.2f} seconds")

#     # Specify Chroma DB path
#     chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
#     logger.debug(f"Using Chroma DB path: {chroma_db_path}")

#     # Check if Chroma DB directory exists
#     if not os.path.exists(chroma_db_path):
#         logger.error(f"Chroma DB path {chroma_db_path} does not exist")
#         raise FileNotFoundError(f"Chroma DB path {chroma_db_path} does not exist")

#     # Check if Chroma DB file exists
#     chroma_db_file = os.path.join(chroma_db_path, "chroma.sqlite3")
#     if not os.path.exists(chroma_db_file):
#         logger.error(f"Chroma DB file not found at {chroma_db_file}")
#         raise FileNotFoundError(f"Chroma DB file not found at {chroma_db_file}")

#     # Initialize Chroma DB
#     logger.debug("Initializing Chroma DB...")
#     start_time = time.time()
#     os.environ["ANONYMIZED_TELEMETRY"] = "False"
#     vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
#     logger.debug(f"Chroma DB initialized in {time.time() - start_time:.2f} seconds")

#     # Initialize LLM
#     logger.debug("Initializing Gemini LLM...")
#     start_time = time.time()
#     google_api_key = os.getenv("GOOGLE_API_KEY")
#     if not google_api_key:
#         logger.error("GOOGLE_API_KEY is not set")
#         raise ValueError("GOOGLE_API_KEY is not set")
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-1.5-flash",
#         temperature=0.7,
#         google_api_key=google_api_key,
#         max_retries=3,
#         timeout=30
#     )
#     logger.debug(f"LLM initialized in {time.time() - start_time:.2f} seconds")

#     # Set up memory and retriever
#     logger.debug("Setting up memory and retriever...")
#     start_time = time.time()
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
#     logger.debug(f"Memory and retriever set up in {time.time() - start_time:.2f} seconds")

#     # Define prompt template
#     prompt_template = """
#     You are a friendly personal assistant with detailed knowledge about Wesley based on the provided documents. Use the context and chat history to answer the user's question accurately, naturally, and helpfully. Always prioritize the most relevant document for the query type.

#     ### Instructions:
#     1. **Personal Details**: Use 'cv.txt' for questions about Wesley’s personal details.
#     2. **Qualifications and Education**: Use 'cv.txt' for qualifications or education.
#     3. **Skills and Work Experience**: Use 'cv.txt' and 'work.txt' for skills or experience.
#     4. **Image Queries**: Select the most relevant image from 'images.txt'.
#     5. **Hobbies and Activities**: Use 'hobbies.txt', 'music.txt', or 'certificates.txt'.
#     6. **Tone and Style**: Speak fondly of Wesley, keep responses concise (2–3 sentences), and avoid speculation.

#     ### Chat History:
#     {chat_history}

#     ### Context:
#     {context}

#     ### Question:
#     {question}

#     ### Answer:
#     """
#     logger.debug("Creating prompt template...")
#     prompt = PromptTemplate(
#         template=prompt_template,
#         input_variables=["chat_history", "context", "question"]
#     )
#     logger.debug("Prompt template created")

#     # Create conversational chain
#     logger.debug("Creating conversational chain...")
#     start_time = time.time()
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         memory=memory,
#         return_source_documents=False,
#         combine_docs_chain_kwargs={"prompt": prompt}
#     )
#     logger.debug(f"Conversational chain created in {time.time() - start_time:.2f} seconds")
#     logger.info("Backend initialization completed successfully")

# except Exception as e:
#     logger.error(f"Backend initialization failed: {str(e)}", exc_info=True)
#     raise

# @app.route('/health')
# def health():
#     logger.debug("Health check endpoint called")
#     return jsonify({'status': 'ok'}), 200

# @app.route('/chat', methods=['POST'])
# def chat():
#     start_time = time.time()
#     logger.debug(f"Received /chat request with headers: {request.headers}")
#     logger.debug(f"Request JSON: {request.json}")
    
#     try:
#         question = request.json.get('question', '')
#         logger.debug(f"Question received: {question}")
#         if not question:
#             logger.warning("No question provided in request")
#             return jsonify({'error': 'No question provided'}), 400

#         # Session initialization
#         logger.debug("Checking session for used_images")
#         if 'used_images' not in session:
#             logger.debug("Initializing session.used_images")
#             session['used_images'] = []

#         # Document retrieval
#         logger.debug(f"Retrieving documents for question: {question}")
#         retrieval_start = time.time()
#         try:
#             docs = retriever.get_relevant_documents(question)  # Revert to older API
#             retrieval_time = time.time() - retrieval_start
#             logger.info(f"Retrieved {len(docs)} documents in {retrieval_time:.2f} seconds")
#             for i, doc in enumerate(docs):
#                 logger.debug(f"Document {i+1}: metadata={doc.metadata}, content={doc.page_content[:100]}...")
#         except Exception as e:
#             logger.error(f"Document retrieval error: {str(e)}", exc_info=True)
#             return jsonify({'error': 'Document retrieval failed'}), 500

#         # Conversation chain invocation
#         logger.debug("Invoking conversation chain...")
#         chain_start = time.time()
#         try:
#             result = conversation_chain.invoke({"question": question})
#             chain_time = time.time() - chain_start
#             logger.info(f"Conversation chain invocation took {chain_time:.2f} seconds")
#             logger.debug(f"Conversation chain result: {result}")
#             answer = result.get('answer', 'Sorry, I couldn’t find an answer.')
#             logger.debug(f"Generated answer: {answer[:100]}...")
#         except Exception as e:
#             logger.error(f"Conversation chain invocation error: {str(e)}", exc_info=True)
#             return jsonify({'error': 'Conversation chain failed'}), 500

#         # Image processing
#         image_url = None
#         if any(keyword in question.lower() for keyword in ["image", "picture", "photo"]):
#             logger.debug("Processing image query")
#             start_image = time.time()
#             try:
#                 for doc in docs:
#                     if 'images.txt' in doc.metadata.get('source', '') and "/images/" in doc.page_content:
#                         image_name = doc.page_content.split("/images/")[1].split()[0].strip(',.')
#                         description = doc.page_content.split(':')[0].lower().strip()
#                         keywords = description.split() + [image_name.lower().replace('.jpg', '').replace('.png', '')]
#                         logger.debug(f"Checking image: {image_name}, keywords: {keywords}")
#                         if any(keyword in question.lower() for keyword in keywords):
#                             if image_name not in session.get('used_images', []):
#                                 image_url = f"https://vector-bot-frontend.vercel.app/images/{image_name}"
#                                 session['used_images'].append(image_name)
#                                 logger.info(f"Selected image: {image_url}")
#                                 break

#                 if not image_url:
#                     logger.debug("No matching image found, trying fallback")
#                     for doc in docs:
#                         if 'images.txt' in doc.metadata.get('source', '') and "/images/" in doc.page_content:
#                             image_name = doc.page_content.split("/images/")[1].split()[0].strip(',.')
#                             if image_name not in session.get('used_images', []):
#                                 image_url = f"https://vector-bot-frontend.vercel.app/images/{image_name}"
#                                 session['used_images'].append(image_name)
#                                 logger.info(f"Fallback image: {image_url}")
#                                 break

#                 if not image_url:
#                     logger.debug("Resetting used_images for image selection")
#                     session['used_images'] = []
#                     for doc in docs:
#                         if 'images.txt' in doc.metadata.get('source', '') and "/images/" in doc.page_content:
#                             image_name = doc.page_content.split("/images/")[1].split()[0].strip(',.')
#                             image_url = f"https://vector-bot-frontend.vercel.app/images/{image_name}"
#                             session['used_images'].append(image_name)
#                             logger.info(f"Reset and selected image: {image_url}")
#                             break
#                 logger.debug(f"Image processing took {time.time() - start_image:.2f} seconds")
#             except Exception as e:
#                 logger.error(f"Image processing error: {str(e)}", exc_info=True)

#         # Prepare response
#         response = {'answer': answer}
#         if image_url:
#             response['image_url'] = image_url
#         logger.debug(f"Prepared response: {response}")

#         # Log total time
#         total_time = time.time() - start_time
#         logger.info(f"Total time for /chat request: {total_time:.2f} seconds")
#         return jsonify(response)

#     except Exception as e:
#         total_time = time.time() - start_time
#         logger.error(f"Unexpected error in /chat after {total_time:.2f} seconds: {str(e)}", exc_info=True)
#         return jsonify({'error': str(e)}), 500

# @app.route('/clear-session', methods=['GET'])
# def clear_session():
#     start_time = time.time()
#     logger.debug("Clearing session")
#     try:
#         session.clear()
#         logger.debug(f"Session cleared in {time.time() - start_time:.2f} seconds")
#         return jsonify({'message': 'Session cleared'})
#     except Exception as e:
#         logger.error(f"Error in /clear-session after {time.time() - start_time:.2f} seconds: {str(e)}", exc_info=True)
#         return jsonify({'error': str(e)}), 500

# @app.route('/')
# def index():
#     logger.debug("Root endpoint called")
#     return jsonify({'message': 'Welcome to Vector Bot Backend. Use /chat for queries.'}), 200

# @app.route('/debug')
# def debug():
#     logger.debug("Debug endpoint called")
#     return jsonify({
#         'cwd': os.getcwd(),
#         'chroma_path': os.path.abspath(chroma_db_path),
#         'chroma_exists': os.path.exists(chroma_db_path),
#         'chroma_file_exists': os.path.exists(chroma_db_file),
#         'data_path': os.path.abspath("./data"),
#         'data_exists': os.path.exists("./data"),
#         'root_contents': os.listdir(os.getcwd())
#     })

# if __name__ == '__main__':
#     port = int(os.getenv("PORT", 5000))
#     logger.info(f"Starting Flask app on port {port}")
#     app.run(host='0.0.0.0', port=port, debug=True)





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

logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://vector-bot-frontend.vercel.app"]}})
app.config["DEBUG"] = True
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")
logger.info("Flask app initialized")

try:
    logger.debug("Loading HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    logger.debug(f"Using Chroma DB path: {chroma_db_path}")
    if not os.path.exists(chroma_db_path):
        logger.error(f"Chroma DB path {chroma_db_path} does not exist")
        raise FileNotFoundError(f"Chroma DB path {chroma_db_path} does not exist")
    
    logger.debug("Initializing Chroma DB...")
    vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
    
    logger.debug("Initializing Gemini LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        max_retries=3,
        timeout=30
    )
    
    logger.debug("Setting up memory and retriever...")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    
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
    prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history", "context", "question"])
    
    logger.debug("Creating conversational chain...")
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
except Exception as e:
    logger.error(f"Backend initialization failed: {str(e)}", exc_info=True)
    raise

@app.route('/health')
def health():
    logger.debug("Health check endpoint called")
    return jsonify({'status': 'ok'}), 200

@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()
    try:
        logger.debug(f"Received /chat request: {request.json}")
        question = request.json.get('question', '')
        if not question:
            logger.warning("No question provided")
            return jsonify({'error': 'No question provided'}), 400

        if 'used_images' not in session:
            logger.debug("Initializing session.used_images")
            session['used_images'] = []

        logger.debug(f"Retrieving documents for question: {question}")
        docs = retriever.invoke(question)
        logger.info(f"Retrieved {len(docs)} documents in {time.time() - start_time:.2f}s")

        logger.debug("Invoking conversation chain...")
        result = conversation_chain.invoke({"question": question})
        answer = result.get('answer', 'Sorry, I couldn’t find an answer.')
        logger.info(f"Answer generated in {time.time() - start_time:.2f}s")

        image_url = None
        if "image" in question.lower() or "picture" in question.lower():
            logger.debug("Processing image query")
            for doc in docs:
                if 'images.txt' in doc.metadata.get('source', '') and "/images/" in doc.page_content:
                    image_name = doc.page_content.split("/images/")[1].split()[0].strip(',.')
                    description = doc.page_content.split(':')[0].lower().strip()
                    keywords = description.split() + [image_name.lower().replace('.jpg', '').replace('.png', '')]
                    if any(keyword in question.lower() for keyword in keywords):
                        if image_name not in session['used_images']:
                            image_url = f"https://vector-bot-frontend.vercel.app/images/{image_name}"
                            session['used_images'].append(image_name)
                            logger.info(f"Selected image: {image_url}")
                            break

        response = {'answer': answer}
        if image_url:
            response['image_url'] = image_url
        logger.info(f"Returning response in {time.time() - start_time:.2f}s")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in /chat: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/clear-session', methods=['GET'])
def clear_session():
    logger.debug("Clearing session")
    session.clear()
    return jsonify({'message': 'Session cleared'})

@app.route('/debug')
def debug():
    logger.debug("Debug endpoint called")
    return jsonify({
        'cwd': os.getcwd(),
        'chroma_path': os.path.abspath(chroma_db_path),
        'chroma_exists': os.path.exists(chroma_db_path),
        'root_contents': os.listdir(os.getcwd())
    })

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port)