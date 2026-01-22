from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from typing import TypedDict, Optional, List, Dict
from dotenv import load_dotenv
from rapidfuzz import fuzz, process

import os
import csv
import yaml
import logging
import re


load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
PROMPTS_PATH = os.path.join(ASSETS_DIR, "prompt.yml")

try:
    with open(PROMPTS_PATH, "r") as f:
        prompts = yaml.safe_load(f)
except FileNotFoundError:
    logging.warning(f"prompt.yml not found at {PROMPTS_PATH}. Using empty prompts.")
    prompts = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PineconeConfig:
    """Configuration for Pinecone connection."""
    
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "sap-items")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
        self.top_k = int(os.getenv("TOP_K", "25"))
        self.alpha = float(os.getenv("ALPHA", "0.2"))  # Lower alpha = more keyword matching
        
        if not self.api_key:
            logger.warning("PINECONE_API_KEY not set. Pinecone search will not work.")
        
        self._embeddings = None
        self._bm25_encoder = None
        self._pc = None
        self._index = None
    
    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        return self._embeddings
    
    @property
    def bm25_encoder(self):
        if self._bm25_encoder is None:
            self._bm25_encoder = BM25Encoder().default()
        return self._bm25_encoder
    
    @property
    def pc(self):
        if self._pc is None and self.api_key:
            self._pc = Pinecone(self.api_key)
        return self._pc
    
    @property
    def index(self):
        if self._index is None and self.pc:
            self._index = self.pc.Index(self.index_name)
        return self._index

pinecone_config = PineconeConfig()


class SAPState(TypedDict):
    item_name: str
    vendor_name: str
    description: str
    categories: List[str]
    fuzzy_item_code: str
    fuzzy_item_name: str
    fuzzy_score: float
    fuzzy_validated: bool
    matched_item_code: str
    matched_item_name: str
    match_method: str
    pinecone_results: List[Dict]

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

model = init_chat_model("google_genai:gemini-2.5-flash-lite", api_key=api_key, max_tokens=500)




def item_description(state: SAPState):
    """Generate a detailed description for an item based on its name and vendor."""
    item_name = state.get("item_name", "")
    vendor_name = state.get("vendor_name", "")

    system_prompt = prompts.get("item_description", {}).get("system", 
        "You are an expert at describing products and items based on their names and vendor context.")
    user_prompt = prompts.get("item_description", {}).get("user", 
        "Describe this item: {item_name} from vendor: {vendor_name}").format(
            item_name=item_name, vendor_name=vendor_name
        )

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    return {
        "description": response.content
    }


def fuzzy_match_csv(state: SAPState) -> dict:
    """Find similar items from item_list.csv using fuzzy matching."""
    item_name = state.get("item_name", "").lower()
    item_names = []
    items = []
    
    item_list_path = os.path.join(ASSETS_DIR, "item_list.csv")
    
    try:
        with open(item_list_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append(row)
                item_names.append(row["ItemName"].lower())
        
        if not item_names:
            logger.warning("item_list.csv is empty")
            return {
                "fuzzy_item_code": "NO_MATCH",
                "fuzzy_item_name": "NO_MATCH",
                "fuzzy_score": 0.0
            }
        
        match = process.extractOne(
            item_name,
            item_names,
            scorer=fuzz.WRatio
        )

        if match:
            logger.info(f"Fuzzy Match - Item: '{match[0]}', Score: {match[1]}, Index: {match[2]}")
            
            return {
                "fuzzy_item_code": items[match[2]]["ItemCode"],
                "fuzzy_item_name": items[match[2]]["ItemName"],
                "fuzzy_score": match[1]
            }
        else:
            return {
                "fuzzy_item_code": "NO_MATCH",
                "fuzzy_item_name": "NO_MATCH",
                "fuzzy_score": 0.0
            }
        
    except FileNotFoundError:
        logger.error(f"item_list.csv not found at {item_list_path}")
        return {
            "fuzzy_item_code": "NO_MATCH",
            "fuzzy_item_name": "NO_MATCH",
            "fuzzy_score": 0.0
        }
    except KeyError as e:
        logger.error(f"Column {e} not found in CSV")
        return {
            "fuzzy_item_code": "NO_MATCH",
            "fuzzy_item_name": "NO_MATCH",
            "fuzzy_score": 0.0
        }
