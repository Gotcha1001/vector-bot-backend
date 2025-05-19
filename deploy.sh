#!/bin/bash
pip install -r requirements.txt
python scripts/create_vector_store.py
gunicorn --chdir backend --bind 0.0.0.0:$PORT app:app