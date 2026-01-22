from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from typing import TypedDict, Optional, List, Dict
from dotenv import load_dotenv
from rapidfuzz import fuzz, process
from google.oauth2 import service_account
from langchain_google_genai import ChatGoogleGenerativeAI

import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.config import settings
import csv
import yaml
import logging
import re
import json

load_dotenv()

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
PROMPTS_PATH = os.path.join(ASSETS_DIR, "prompt.yml")

# Load prompts
try:
    with open(PROMPTS_PATH, "r") as f:
        prompts = yaml.safe_load(f)
except FileNotFoundError:
    logging.warning(f"prompt.yml not found at {PROMPTS_PATH}. Using empty prompts.")
    prompts = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CATEGORY_TO_NAMESPACE = {
    "Electrical": "electrical",
    "Mechanicals": "mechanical",
    "Civil": "civil",
    "IT & Accessories": "it_accessories",
    "Heavy Equipments & Automobiles": "heavy_equipments_automobiles",
    "Business Promotion & Marketing": "business_promotion_marketing",
    "Finished Goods": "finished_goods",
    "Raw Material": "raw_material",
    "Auxiliary Raw Material": "auxiliary_raw_material",
    "Printing & Stationary": "printing_stationary",
    "Other": "other",
    "Asset": "asset",
    "Health & Safety": "health_safety",
    "Fuel Lubricant and Gas": "fuel_lubricant_and_gas",
    "Mesh": "mesh",
    "Trading": "trading",
    "Wastages": "wastages",
    "Service": "service",
}


class PineconeMapperState(TypedDict):
    """State for the Pinecone item name mapper graph."""
    item_name: str
    vendor_name: str
    description: str
    categories: List[str]  # List of categories from LLM
    fuzzy_item_code: str
    fuzzy_item_name: str
    fuzzy_item_uom_group_entry: str  # UoMGroupEntry from CSV fuzzy match
    fuzzy_item_inventory_uom_entry: str  # InventoryUoMEntry from CSV fuzzy match
    fuzzy_score: float
    pinecone_results: List[Dict]  # List of {item_code, item_name, score, namespace, uom_group, inventory_uom}
    matched_item_code: str
    matched_item_name: str
    matched_item_uom_group_entry: str  # UoMGroupEntry from final match
    matched_item_inventory_uom_entry: str  # InventoryUoMEntry from final match
    match_method: str
    fuzzy_validated: bool


class PineconeConfig:
    """Configuration for Pinecone connection."""
    
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "sap-items")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
        self.top_k = int(os.getenv("TOP_K", "25"))
        self.alpha = float(os.getenv("ALPHA", "0.2"))  
        
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


credentials = service_account.Credentials.from_service_account_file(
    settings.GOOGLE_APPLICATION_CREDENTIALS.get_secret_value(),
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    project=settings.GOOGLE_CLOUD_PROJECT.get_secret_value(),
    credentials=credentials,
    location=settings.GOOGLE_CLOUD_LOCATION,
)

def get_namespaces_from_categories(categories: List[str]) -> List[str]:
    """Convert category names to Pinecone namespace names."""
    namespaces = []
    for category in categories:
        category = category.strip()
        # Try direct mapping
        if category in CATEGORY_TO_NAMESPACE:
            namespaces.append(CATEGORY_TO_NAMESPACE[category])
        else:
            # Try case-insensitive matching
            for cat_name, namespace in CATEGORY_TO_NAMESPACE.items():
                if cat_name.lower() == category.lower():
                    namespaces.append(namespace)
                    break
            else:
                # Convert category name to namespace format
                namespace = category.lower().replace(" ", "_").replace("&", "and")
                namespaces.append(namespace)
    
    return list(set(namespaces))  # Remove duplicates


def parse_categories_from_llm_response(response: str) -> List[str]:
    """Parse categories from LLM response (comma-separated)."""
    # Remove any markdown formatting
    response = response.strip()
    
    # Split by comma and clean up
    categories = [cat.strip() for cat in response.split(",")]
    
    # Remove empty strings
    categories = [cat for cat in categories if cat]
    
    return categories


def generate_item_description(state: PineconeMapperState) -> dict:
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
    
    logger.info(f"Generated description for '{item_name}'")
    
    return {
        "description": response.content
    }


def fuzzy_match_csv(state: PineconeMapperState) -> dict:
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
                "fuzzy_item_uom_group_entry": "",
                "fuzzy_item_inventory_uom_entry": "",
                "fuzzy_score": 0.0
            }
        
        match = process.extractOne(
            item_name,
            item_names,
            scorer=fuzz.WRatio
        )

        if match:
            logger.info(f"Fuzzy Match - Item: '{match[0]}', Score: {match[1]}, Index: {match[2]}")
            matched_item = items[match[2]]
            
            return {
                "fuzzy_item_code": matched_item.get("ItemCode", ""),
                "fuzzy_item_name": matched_item.get("ItemName", ""),
                "fuzzy_item_uom_group_entry": matched_item.get("UoMGroupEntry", ""),
                "fuzzy_item_inventory_uom_entry": matched_item.get("InventoryUoMEntry", ""),
                "fuzzy_score": match[1]
            }
        else:
            return {
                "fuzzy_item_code": "NO_MATCH",
                "fuzzy_item_name": "NO_MATCH",
                "fuzzy_item_uom_group_entry": "",
                "fuzzy_item_inventory_uom_entry": "",
                "fuzzy_score": 0.0
            }
        
    except FileNotFoundError:
        logger.error(f"item_list.csv not found at {item_list_path}")
        return {
            "fuzzy_item_code": "NO_MATCH",
            "fuzzy_item_name": "NO_MATCH",
            "fuzzy_item_uom_group_entry": "",
            "fuzzy_item_inventory_uom_entry": "",
            "fuzzy_score": 0.0
        }
    except KeyError as e:
        logger.error(f"Column {e} not found in CSV")
        return {
            "fuzzy_item_code": "NO_MATCH",
            "fuzzy_item_name": "NO_MATCH",
            "fuzzy_item_uom_group_entry": "",
            "fuzzy_item_inventory_uom_entry": "",
            "fuzzy_score": 0.0
        }


def check_fuzzy_match(state: PineconeMapperState) -> str:
    """Conditional: Check if fuzzy match score is good enough."""
    fuzzy_score = state.get("fuzzy_score", 0.0)
    
    if fuzzy_score >= 97:
        logger.info(f"Fuzzy score {fuzzy_score} >= 97, validating with LLM")
        return "validate_fuzzy_match"
    else:
        logger.info(f"Fuzzy score {fuzzy_score} < 97, proceeding to category mapping")
        return "categorize_item"


def validate_fuzzy_match_with_llm(state: PineconeMapperState) -> str:
    """Conditional: Validate fuzzy match using LLM."""
    item_name = state.get("item_name", "")
    description = state.get("description", "")
    fuzzy_item_name = state.get("fuzzy_item_name", "")

    system_prompt = prompts.get("fuzzy_itemname_validation", {}).get("system",
        """You are an expert at determining if two product names refer to the same item.
        You must respond with ONLY 'true' or 'false'. Do not include any explanation.""")
    
    user_prompt = prompts.get("fuzzy_itemname_validation", {}).get("user",
        """Determine if the fuzzy matched item is the same as the original item.
        
        Original Item Name: '{item_name}'
        Original Item Description: '{item_description}'
        Fuzzy Matched Item Name: '{fuzzy_item_name}'
        
        Respond with only 'true' or 'false'.""").format(
            item_name=item_name,
            item_description=description,
            fuzzy_item_name=fuzzy_item_name
        )

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    result = response.content.strip().lower()
    logger.info(f"LLM validation of fuzzy match '{item_name}' -> '{fuzzy_item_name}': {result}")
    
    if "true" in result:
        return "accept_fuzzy_match"
    else:
        return "categorize_item"


def accept_fuzzy_match(state: PineconeMapperState) -> dict:
    """Accept the fuzzy match result including UoM fields."""
    return {
        "matched_item_code": state.get("fuzzy_item_code", "NO_MATCH"),
        "matched_item_name": state.get("fuzzy_item_name", ""),
        "matched_item_uom_group_entry": state.get("fuzzy_item_uom_group_entry", ""),
        "matched_item_inventory_uom_entry": state.get("fuzzy_item_inventory_uom_entry", ""),
        "match_method": "fuzzy_validated",
        "fuzzy_validated": True
    }


def categorize_item(state: PineconeMapperState) -> dict:
    """Use LLM to categorize the item into one or more categories."""
    item_name = state.get("item_name", "")
    description = state.get("description", "")

    system_prompt = prompts.get("category_mapping", {}).get("system",
        "You are an expert at categorizing products and items.")
    
    user_prompt = prompts.get("category_mapping", {}).get("user",
        "Categorize this item: {item_name}. Description: {description}").format(
            item_name=item_name, description=description
        )

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    # Parse categories from response
    categories = parse_categories_from_llm_response(response.content)
    
    logger.info(f"LLM categorized '{item_name}' into: {categories}")
    
    return {
        "categories": categories
    }


def search_pinecone_by_categories(state: PineconeMapperState) -> dict:
    """
    Search Pinecone for similar items in the categorized namespaces.
    Returns items with their ItemCode from metadata.
    """
    item_name = state.get("item_name", "")
    description = state.get("description", "")
    categories = state.get("categories", [])
    
    if not pinecone_config.api_key:
        logger.error("Pinecone API key not configured")
        return {"pinecone_results": []}
    
    # Convert categories to namespaces
    namespaces = get_namespaces_from_categories(categories)
    
    if not namespaces:
        logger.warning("No valid namespaces found for categories")
        return {"pinecone_results": []}
    
    # Combine item name and description for better search
    query = f"{item_name} {description[:200]}" if description else item_name
    
    all_results = []
    
    for namespace in namespaces:
        try:
            logger.info(f"Searching Pinecone namespace: {namespace}")
            
            retriever = PineconeHybridSearchRetriever(
                embeddings=pinecone_config.embeddings,
                sparse_encoder=pinecone_config.bm25_encoder,
                index=pinecone_config.index,
                namespace=namespace,
                top_k=pinecone_config.top_k,
                alpha=pinecone_config.alpha,
                text_key="text",
            )
            
            results = retriever.invoke(query)
            
            for doc in results:
                # Extract item_code and UoM fields from metadata
                metadata = doc.metadata or {}
                item_code = metadata.get("item_code") or metadata.get("ItemCode") or metadata.get("code") or ""
                score = metadata.get("score", 0.0)
                
                # Extract UoM fields - try different possible key names
                uom_group_entry = (metadata.get("UoMGroupEntry"))
                inventory_uom_entry = (metadata.get("InventoryUoMEntry"))

                result_item = {
                    "item_code": item_code,
                    "item_name": doc.page_content,
                    "score": score,
                    "namespace": namespace,
                    "uom_group_entry": uom_group_entry,
                    "inventory_uom_entry": inventory_uom_entry,
                    "metadata": metadata
                }
                all_results.append(result_item)
                
            logger.info(f"Found {len(results)} results in namespace '{namespace}'")
            
        except Exception as e:
            logger.error(f"Error searching namespace '{namespace}': {e}")
            continue
    
    # Sort by score (if available) or keep order
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    logger.info(f"Total Pinecone results: {len(all_results)}")
    
    return {
        "pinecone_results": all_results
    }


def match_with_llm_from_pinecone(state: PineconeMapperState) -> dict:
    """
    Use LLM to select the best match from Pinecone results.
    The LLM receives both item names AND item codes.
    """
    item_name = state.get("item_name", "")
    description = state.get("description", "")
    vendor_name = state.get("vendor_name", "")
    categories = state.get("categories", [])
    pinecone_results = state.get("pinecone_results", [])
    
    if not pinecone_results:
        logger.warning("No Pinecone results to match against")
        return {
            "matched_item_code": "NO_MATCH",
            "matched_item_name": "",
            "match_method": "no_pinecone_results"
        }
    
    # Format items list with BOTH code and name
    # This is the key change - include item_code in the list
    items_list = "\n".join([
        f"    {item['item_code']}: {item['item_name']}" 
        for item in pinecone_results 
        if item.get('item_code')  # Only include items with codes
    ])
    
    if not items_list:
        # Fallback if no items have codes - use item names only
        items_list = "\n".join([
            f"    {item['item_name']}" 
            for item in pinecone_results
        ])
        logger.warning("No item codes found in Pinecone results, using names only")
    
    system_prompt = prompts.get("item_mapping_through_category", {}).get("system",
        "You are an expert at matching items to their SAP codes.")
    
    user_prompt = prompts.get("item_mapping_through_category", {}).get("user",
        """Match this item to the correct SAP item code:
        Item Name: {item_name}
        Description: {description}
        Vendor: {vendor_name}
        Category: {category}
        
        Available items (format: ItemCode: ItemName):
        {items_list}
        
        Return ONLY the item code if you find a match, or 'NO_MATCH' if none match.
        Do not include any explanation.""").format(
            item_name=item_name,
            description=description,
            vendor_name=vendor_name,
            category=", ".join(categories),
            items_list=items_list
        )
    
    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    matched_code = response.content.strip()
    
    # Clean up the response - extract just the code
    # Handle cases like "FG00971" or "The best match is FG00971"
    code_match = re.search(r'\b([A-Z]{2}\d+)\b', matched_code.upper())
    if code_match:
        matched_code = code_match.group(1)
    
    logger.info(f"LLM matched '{item_name}' to: {matched_code}")
    
    # Find the matched item details from results
    matched_name = ""
    matched_uom_group_entry = ""
    matched_inventory_uom_entry = ""
    
    if matched_code and matched_code != "NO_MATCH":
        for item in pinecone_results:
            if item.get("item_code", "").upper() == matched_code.upper():
                matched_name = item.get("item_name", "")
                matched_uom_group_entry = item.get("uom_group_entry", "")
                matched_inventory_uom_entry = item.get("inventory_uom_entry", "")
                break
    
    return {
        "matched_item_code": matched_code if matched_code != "NO_MATCH" else "NO_MATCH",
        "matched_item_name": matched_name,
        "matched_item_uom_group_entry": matched_uom_group_entry,
        "matched_item_inventory_uom_entry": matched_inventory_uom_entry,
        "match_method": "pinecone_llm_match" if matched_code != "NO_MATCH" else "no_match"
    }


"""
Flow:
START -> generate_item_description -> fuzzy_match_csv -> check_fuzzy_match
    ├── (score >= 97) -> validate_fuzzy_match_with_llm
    │     ├── (true)  -> accept_fuzzy_match -> END
    │     └── (false) -> categorize_item -> search_pinecone -> match_with_llm -> END
    │
    └── (score < 97)  -> categorize_item -> search_pinecone -> match_with_llm -> END
"""

graph = StateGraph(PineconeMapperState)


graph.add_node("generate_description", generate_item_description)
graph.add_node("fuzzy_match", fuzzy_match_csv)
graph.add_node("accept_fuzzy_match", accept_fuzzy_match)
graph.add_node("categorize_item", categorize_item)
graph.add_node("search_pinecone", search_pinecone_by_categories)
graph.add_node("match_with_llm", match_with_llm_from_pinecone)

graph.add_edge(START, "generate_description")
graph.add_edge("generate_description", "fuzzy_match")

# Conditional: check fuzzy score
graph.add_conditional_edges(
    "fuzzy_match",
    check_fuzzy_match,
    {
        "validate_fuzzy_match": "validate_fuzzy_match", 
        "categorize_item": "categorize_item"
    }
)


# We need to add it as a node first, then add conditional edges from it
graph.add_node("validate_fuzzy_match", lambda state: state) 

graph.add_conditional_edges(
    "validate_fuzzy_match",
    validate_fuzzy_match_with_llm,
    {
        "accept_fuzzy_match": "accept_fuzzy_match",
        "categorize_item": "categorize_item"
    }
)

# accept_fuzzy_match -> END
graph.add_edge("accept_fuzzy_match", END)

# categorize_item -> search_pinecone -> match_with_llm -> END
graph.add_edge("categorize_item", "search_pinecone")
graph.add_edge("search_pinecone", "match_with_llm")
graph.add_edge("match_with_llm", END)

app = graph.compile()


# processing items

def process_item(item_name: str, vendor_name: str = "") -> Dict:
    """
    Process a single item and return the matched SAP item code.
    
    Args:
        item_name: The item name from OCR/document
        vendor_name: The vendor name from OCR/document
    
    Returns:
        Dictionary with matched item code and metadata
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {item_name} (Vendor: {vendor_name})")
    logger.info(f"{'='*60}")
    
    # Initialize state
    initial_state = {
        "item_name": item_name,
        "vendor_name": vendor_name,
        "description": "",
        "categories": [],
        "fuzzy_item_code": "NO_MATCH",
        "fuzzy_item_name": "",
        "fuzzy_item_uom_group_entry": "",
        "fuzzy_item_inventory_uom_entry": "",
        "fuzzy_score": 0.0,
        "pinecone_results": [],
        "matched_item_code": "NO_MATCH",
        "matched_item_name": "",
        "matched_item_uom_group_entry": "",
        "matched_item_inventory_uom_entry": "",
        "match_method": "",
        "fuzzy_validated": False
    }
    
    # Run the graph
    result = app.invoke(initial_state)
    
    logger.info(f"Result: {item_name} -> {result.get('matched_item_code')} ({result.get('match_method')})")
    
    return {
        "item_name": item_name,
        "vendor_name": vendor_name,
        "description": result.get("description", ""),
        "categories": result.get("categories", []),
        "fuzzy_item_code": result.get("fuzzy_item_code", ""),
        "fuzzy_item_name": result.get("fuzzy_item_name", ""),
        "fuzzy_item_uom_group_entry": result.get("fuzzy_item_uom_group_entry", ""),
        "fuzzy_item_inventory_uom_entry": result.get("fuzzy_item_inventory_uom_entry", ""),
        "fuzzy_score": result.get("fuzzy_score", 0.0),
        "fuzzy_validated": result.get("fuzzy_validated", False),
        "matched_item_code": result.get("matched_item_code", "NO_MATCH"),
        "matched_item_name": result.get("matched_item_name", ""),
        "matched_item_uom_group_entry": result.get("matched_item_uom_group_entry", ""),
        "matched_item_inventory_uom_entry": result.get("matched_item_inventory_uom_entry", ""),
        "match_method": result.get("match_method", "")
    }


def process_ocr_items(ocr_result: dict) -> List[Dict]:
    """
    Process line items from OCR result and map them to SAP item codes.
    
    Args:
        ocr_result: The OCR result dictionary containing vendor_details and line_items
    
    Returns:
        List of mapped items with their SAP codes
    """
    # Extract vendor name from OCR result
    vendor_name = ocr_result.get("vendor_details", {}).get("name", "")
    
    # Extract line items - 'products' field contains the item description/name
    line_items = ocr_result.get("line_items", [])
    
    if not line_items:
        logger.warning("No line items found in OCR result")
        return []
    
    mapped_items = []
    
    for item in line_items:
        # Get item name from 'products' field (as per OCR processor schema)
        item_name = item.get("products", "") or item.get("description", "")
        
        if not item_name:
            logger.warning(f"Skipping item with no name: {item}")
            continue
        
        # Process the item
        result = process_item(item_name, vendor_name)
        
        # Add original item data
        result["original_item"] = item
        
        mapped_items.append(result)
    
    return mapped_items


#test
if __name__ == "__main__":
    # Test with sample items
    test_items = [
        {"item_name": "SPRING Washer B-8", "vendor_name": "Hardware Supplies"},
        {"item_name": "billet small length", "vendor_name": "Steel Corp"},

    ]
    
    print("="*80)
    print("TESTING PINECONE ITEM NAME MAPPER")
    print("="*80)
    
    for test in test_items:
        result = process_item(test["item_name"], test["vendor_name"])
        
        print(f"\n{'─'*60}")
        print(f"Input: {test['item_name']} (Vendor: {test['vendor_name']})")
        print(f"Description: {result['description'][:80]}..." if len(result.get('description', '')) > 80 else f"Description: {result.get('description', '')}")
        print(f"Categories: {', '.join(result.get('categories', []))}")
        print(f"Fuzzy Match: {result['fuzzy_item_code']} - {result['fuzzy_item_name']} (Score: {result['fuzzy_score']:.1f})")
        print(f"  └─ Fuzzy UoM Group: {result['fuzzy_item_uom_group_entry']}, Inventory UoM: {result['fuzzy_item_inventory_uom_entry']}")
        print(f"Fuzzy Validated: {result['fuzzy_validated']}")
        print(f"Final Match: {result['matched_item_code']} - {result['matched_item_name']}")
        print(f"  └─ UoM Group Entry: {result['matched_item_uom_group_entry']}")
        print(f"  └─ Inventory UoM Entry: {result['matched_item_inventory_uom_entry']}")
        print(f"Match Method: {result['match_method']}")
