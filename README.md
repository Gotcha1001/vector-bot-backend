# Vector Bot Backend

A Flask-based backend for the Vector Bot application, using LangChain and Chroma for conversational AI.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment variables in `.env`
3. Run the vector store script: `python scripts/create_vector_store.py`
4. Start the server: `python backend/app.py`

## Endpoints

- `/health`: Check server status
- `/chat`: Handle chat queries (POST)
- `/clear-session`: Clear session data (GET)
