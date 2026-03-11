import os
from dotenv import load_dotenv
# from pymongo import MongoClient
from pymongo import AsyncMongoClient
from .core.config import settings
from fastapi import Request

load_dotenv()

mongodb_uri = settings.MONGODB_URI

client = AsyncMongoClient(mongodb_uri)
db = client["sap_ocr"]
collection = db["documents"]


async def check_collection():
    if "documents" not in await db.list_collection_names():
        await db.create_collection("documents")
        
    doc_collection = db["documents"]
    
    return doc_collection
    
async def add_default_prompt(prompt):
    collection = await check_collection()
    
    if await collection.count_documents({"default_type": "pdf"}) == 0:
        await collection.insert_one({"default_type": "pdf", "default_prompt": prompt})
        
async def init_db(request: Request):
    return request.app.state.collection
    