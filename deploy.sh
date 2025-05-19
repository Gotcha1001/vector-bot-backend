#!/bin/bash
# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Ensure Chroma DB directory exists and is writable
chroma_db_path=${CHROMA_DB_PATH:-/opt/render/project/chroma_db}
echo "Setting up $chroma_db_path..."
mkdir -p $chroma_db_path
chmod -R 777 $chroma_db_path
ls -ld $chroma_db_path
echo "Current user: $(whoami)"
echo "Directory permissions: $(stat -c '%A %U:%G' $chroma_db_path)"

# Create or verify Chroma DB
echo "Running create_vector_store.py..."
python scripts/create_vector_store.py

# Start Gunicorn
echo "Starting Gunicorn..."
gunicorn --chdir backend --bind 0.0.0.0:$PORT app:app