import os
from dotenv import load_dotenv
from pymongo import MongoClient
from .core.config import settings

load_dotenv()

mongodb_uri = settings.MONGODB_URI

client = MongoClient(mongodb_uri)
db = client["sap_ocr"]
collection = db["documents"]
    
if "documents" not in db.list_collection_names():
    db.create_collection("documents")

def add_default_prompt(prompt):
    if collection.count_documents({"default_type": "pdf"}) == 0:
        collection.insert_one({"default_type": "pdf", "default_prompt": prompt})

    