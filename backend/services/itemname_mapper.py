from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from typing import TypedDict, Optional
from neo4j import GraphDatabase
from dotenv import load_dotenv
from rapidfuzz import fuzz, process

import os
import csv
import yaml
import logging

load_dotenv()

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


# Mapping from item code prefix to category names
PREFIX_TO_CATEGORY = {
    "EL": "Electrical",
    "ME": "Mechanicals",
    "CV": "Civil",
    "IT": "IT & Accessories",
    "HE": "Heavy Equipments & Automobiles",
    "BP": "Business Promotion & Marketing",
    "FG": "Finished Goods",  # Could also be Finished Goods - TMT, Billet, or Others
    "RM": "Raw Material",
    "AR": "Auxiliary Raw Material",
    "PS": "Printing & Stationary",
    "OT": "Other",
    "AS": "Asset",
    "HS": "Health & Safety",
    "FL": "Fuel Lubricant and Gas",
    "MS": "Mesh",
    "TR": "Trading",
    "WS": "Wastages",
    "SV": "Service",
}


def get_category_from_item_code(item_code: str) -> str:
    """
    Extract category from item code prefix.
    
    Args:
        item_code: The item code (e.g., 'EL1123', 'CV001')
    
    Returns:
        Category name or None if prefix not found
    """
    if not item_code or item_code == "NO_MATCH":
        return None
    
    # Extract prefix (first 2 characters)
    prefix = item_code[:2].upper()
    category = PREFIX_TO_CATEGORY.get(prefix)
    
    if category:
        logger.info(f"Extracted category '{category}' from item code prefix '{prefix}'")
    else:
        logger.warning(f"Unknown item code prefix: '{prefix}'")
    
    return category


class Neo4jConnection:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"), 
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Neo4j connection is working."""
        try:
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed.")
    

class SAPState(TypedDict):
    item_name: str
    vendor_name: str
    description: str
    category: str
    matched_item_code: str
    matched_item_name: str
    fuzzy_item_code: str
    fuzzy_item_name: str
    fuzzy_validated: bool
    match_method: str


# Neo4j connection singleton for graph nodes
_neo4j_conn: Optional["Neo4jConnection"] = None


def set_neo4j_connection(conn: Optional["Neo4jConnection"]):
    """Set the Neo4j connection for graph nodes to use."""
    global _neo4j_conn
    _neo4j_conn = conn


def get_neo4j_connection() -> Optional["Neo4jConnection"]:
    """Get the Neo4j connection for graph nodes."""
    return _neo4j_conn


# Initialize LLM model
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


def get_similar_item(state: SAPState):
    """Find similar items from the item_list.csv using fuzzy matching."""
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
                "ItemCode": "NO_MATCH",
                "ItemName": "NO_MATCH"
            }
        
        match = process.extractOne(
            item_name,
            item_names,
            scorer=fuzz.WRatio
        )

        logger.info(f"Fuzzy Match - Item: '{match[0]}', Score: {match[1]}, Index: {match[2]}")

        if match and match[1] >= 80:
            matched_item = {
                "ItemCode": items[match[2]]["ItemCode"],
                "ItemName": items[match[2]]["ItemName"]
            }
            logger.info(f"Matched item from CSV: {matched_item['ItemCode']} - {matched_item['ItemName']}")
        else:
            logger.info(f"No match found (score {match[1]} < 80)")
            matched_item = {
                "ItemCode": "NO_MATCH",
                "ItemName": "NO_MATCH"
            }

        return matched_item
        
    except FileNotFoundError:
        logger.error(f"item_list.csv not found at {item_list_path}. Skipping fuzzy matching.")
        return {
            "ItemCode": "NO_MATCH",
            "ItemName": "NO_MATCH"
        }
    except KeyError as e:
        logger.error(f"Column {e} not found in CSV. Expected columns: ItemCode, ItemName")
        return {
            "ItemCode": "NO_MATCH",
            "ItemName": "NO_MATCH"
        }


def fuzzy_conditional_node(state: SAPState) -> str:
    """
    Conditional node to determine next step based on fuzzy match result.
    Returns the name of the next node to execute.
    """
    matched_item = get_similar_item(state)
    
    if matched_item["ItemCode"] != "NO_MATCH":
        # Good fuzzy match found, store result and validate with LLM
        logger.info(f"Fuzzy match found: {matched_item['ItemCode']} - {matched_item['ItemName']}")
        return "store_fuzzy_result"
    else:
        # No match, continue to category mapping
        logger.info("No fuzzy match found, proceeding to category mapping")
        return "category_mapping"


def category_mapping(state: SAPState):
    """Map item to a category using LLM."""
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
    
    return {
        "category": response.content
    }


def get_items_by_category(category: str, neo4j_conn: Neo4jConnection) -> list[dict]:
    """Fetch items from Neo4j by category."""
    all_items = []
    # Split category by comma and clean whitespace
    categories = [cat.strip() for cat in category.split(",")]

    with neo4j_conn.driver.session(database="neo4j") as session:
        for category_name in categories:
            logger.info(f"Fetching items for category: {category_name}")
            result = session.run('''
                MATCH (i:Item)-[:BELONGS_TO]->(c:Category {name: $category_name})
                RETURN i.code AS code, i.name AS name
                ORDER BY i.code
            ''', category_name=category_name)
            
            items = [{"code": record["code"], "name": record["name"], "category": category_name} 
                    for record in result]
            all_items.extend(items)
            logger.info(f"Found {len(items)} items in category '{category_name}'")
    
    logger.info(f"Total items found across all categories: {len(all_items)}")
    return all_items


def validate_fuzzy_itemname_with_llm(state: SAPState) -> str:
    """
    Conditional node: Validate fuzzy matched item name with LLM.
    Returns next node name based on validation result.
    """
    item_name = state.get("item_name", "")
    item_description = state.get("description", "")
    fuzzy_item_name = state.get("fuzzy_item_name", "")

    system_prompt = prompts.get("fuzzy_itemname_validation", {}).get("system",
        """You are an expert at validating if two product names refer to the same item.
        You must respond with only 'true' or 'false'.""")
    user_prompt = prompts.get("fuzzy_itemname_validation", {}).get("user",
        """Determine if the fuzzy matched item is the same as the original item.
        
        Original Item Name: {item_name}
        Original Item Description: {item_description}
        Fuzzy Matched Item Name: {fuzzy_item_name}
        
        Consider:
        - Do they refer to the same type of product?
        - Are the specifications similar (size, grade, type)?
        - Would they be used for the same purpose?
        
        Respond with only 'true' if they match, or 'false' if they don't.""").format(
            item_name=item_name,
            item_description=item_description,      
            fuzzy_item_name=fuzzy_item_name
        )

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    boolean_response = response.content.strip().lower()
    
    logger.info(f"LLM validation of fuzzy match '{item_name}' -> '{fuzzy_item_name}': {boolean_response}")
    
    if "true" in boolean_response:
        return "fetch_item_info_node"
    else:
        return "category_mapping"


def fetch_item_info_node(state: SAPState):
    """
    Graph node: Fetch item info from Neo4j for fuzzy matched item.
    Uses the fuzzy_item_code from state.
    """
    fuzzy_item_code = state.get("fuzzy_item_code", "")
    fuzzy_item_name = state.get("fuzzy_item_name", "")
    
    neo4j_conn = get_neo4j_connection()
    
    if neo4j_conn and fuzzy_item_code:
        item_info = fetch_item_info(fuzzy_item_code, neo4j_conn)
        if item_info:
            return {
                "matched_item_code": item_info["code"],
                "matched_item_name": item_info["name"],
                "category": ", ".join(item_info.get("categories", [])),
                "fuzzy_validated": True,
                "match_method": "fuzzy_llm_validated"
            }
    
    # Fallback to fuzzy result if Neo4j not available
    return {
        "matched_item_code": fuzzy_item_code,
        "matched_item_name": fuzzy_item_name,
        "fuzzy_validated": True,
        "match_method": "fuzzy_validated_no_neo4j"
    }


def get_items_by_category_node(state: SAPState):
    """
    Graph node: Fetch items from Neo4j by category.
    Stores the items in state for the next node.
    """
    category = state.get("category", "")
    neo4j_conn = get_neo4j_connection()
    
    if not neo4j_conn:
        logger.warning("No Neo4j connection available for get_items_by_category")
        return {"category_items": []}
    
    if not category:
        logger.warning("No category provided for get_items_by_category")
        return {"category_items": []}
    
    items = get_items_by_category(category, neo4j_conn)
    
    # Store items as a string representation for state (TypedDict limitation)
    # We'll parse it back in the next node
    return {"category_items_count": len(items)}


def match_item_with_llm_node(state: SAPState):
    """
    Graph node: Match item with LLM using category items from Neo4j.
    """
    item_name = state.get("item_name", "")
    description = state.get("description", "")
    vendor_name = state.get("vendor_name", "")
    category = state.get("category", "")
    
    neo4j_conn = get_neo4j_connection()
    
    if not neo4j_conn:
        logger.warning("No Neo4j connection for LLM matching")
        return {
            "matched_item_code": "NO_MATCH",
            "matched_item_name": "",
            "match_method": "no_neo4j"
        }
    
    # Get items from category
    items = get_items_by_category(category, neo4j_conn)
    
    if not items:
        logger.warning(f"No items found in category '{category}'")
        return {
            "matched_item_code": "NO_MATCH",
            "matched_item_name": "",
            "match_method": "no_category_items"
        }
    
    # Use LLM to find best match
    matched_code = match_item_with_llm(
        item_name=item_name,
        description=description,
        vendor_name=vendor_name,
        category=category,
        items=items
    )
    
    if matched_code and matched_code != "NO_MATCH":
        # Fetch item details
        item_info = fetch_item_info(matched_code, neo4j_conn)
        if item_info:
            return {
                "matched_item_code": matched_code,
                "matched_item_name": item_info.get("name", ""),
                "match_method": "llm_category_match"
            }
        return {
            "matched_item_code": matched_code,
            "matched_item_name": "",
            "match_method": "llm_category_match"
        }
    
    return {
        "matched_item_code": "NO_MATCH",
        "matched_item_name": "",
        "match_method": "no_match"
    }


def store_fuzzy_result(state: SAPState):
    """
    Graph node: Store fuzzy match result in state before validation.
    """
    fuzzy_result = get_similar_item(state)
    
    return {
        "fuzzy_item_code": fuzzy_result.get("ItemCode", "NO_MATCH"),
        "fuzzy_item_name": fuzzy_result.get("ItemName", "NO_MATCH")
    }


def match_item_with_llm(item_name: str, description: str, vendor_name: str, 
                       category: str, items: list[dict]) -> str: 
    """Match an item to a SAP item code using LLM."""
    # Format items list for the prompt
    items_list = "\n    ".join([f"{item['code']}: {item['name']}" for item in items])
    
    system_prompt = prompts.get("item_mapping_through_category", {}).get("system",
        "You are an expert at matching items to their SAP codes.")
    user_prompt = prompts.get("item_mapping_through_category", {}).get("user",
        """Match this item to the correct SAP item code:
        Item Name: {item_name}
        Description: {description}
        Vendor: {vendor_name}
        Category: {category}
        
        Available items:
        {items_list}
        
        Return only the item code.""").format(
            item_name=item_name,
            description=description,
            vendor_name=vendor_name,
            category=category,
            items_list=items_list
        )
    
    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    matched_code = response.content.strip()
    logger.info(f"LLM matched item '{item_name}' to: {matched_code}")
    
    return matched_code


def fetch_item_info(item_code: str, neo4j_conn: Neo4jConnection) -> dict:
    """ Fetch detailed information about an item from Neo4j by its code. """
    
    with neo4j_conn.driver.session(database="neo4j") as session:
        result = session.run('''
            MATCH (i:Item)
            WHERE i.code = $item_code
            OPTIONAL MATCH (i)-[:BELONGS_TO]->(c:Category)
            RETURN i.code AS code, i.name AS name, 
                   collect(c.name) AS categories
        ''', item_code=item_code)
        
        record = result.single()
        if record:
            item_info = {
                "code": record["code"],
                "name": record["name"],
                "categories": record["categories"]
            }
            logger.info(f"Fetched item info for {item_code}: {item_info}")
            return item_info
        else:
            logger.warning(f"Item code {item_code} not found in database")
            return None


def map_category_through_itemcode(state: SAPState):
    """
    Map category through item code prefix.
    Gets the fuzzy matched item code, extracts category from prefix,
    and stores it in state for later validation.
    """
    item_name = state.get("item_name", "")
    
    # Get fuzzy match result
    fuzzy_result = get_similar_item(state)
    item_code = fuzzy_result.get("ItemCode", "NO_MATCH")
    
    # Extract category from item code prefix
    category = get_category_from_item_code(item_code)
    
    if category:
        logger.info(f"Item '{item_name}' fuzzy matched to '{item_code}' -> Category: '{category}'")
        return {
            "category": category,
            "matched_item_code": item_code
        }
    else:
        logger.warning(f"Could not determine category for item code '{item_code}'")
        return {
            "category": "",
            "matched_item_code": item_code
        }


# =============================================================================
# Build the LangGraph
# =============================================================================
# Flow:
# START -> item_description -> fuzzy_conditional_node
#   ├── YES (fuzzy match found) -> store_fuzzy_result -> validate_fuzzy_itemname_with_llm
#   │     ├── TRUE  -> fetch_item_info_node -> END
#   │     └── FALSE -> category_mapping -> match_item_with_llm_node -> END
#   │
#   └── NO (no fuzzy match) -> category_mapping -> match_item_with_llm_node -> END
# =============================================================================

graph = StateGraph(SAPState)

# Add all nodes
graph.add_node("item_description", item_description)
graph.add_node("store_fuzzy_result", store_fuzzy_result)
graph.add_node("category_mapping", category_mapping)
graph.add_node("fetch_item_info_node", fetch_item_info_node)
graph.add_node("match_item_with_llm_node", match_item_with_llm_node)

# START -> item_description
graph.add_edge(START, "item_description")

# item_description -> fuzzy_conditional_node (conditional)
# If fuzzy match found -> store_fuzzy_result
# If no match -> category_mapping
graph.add_conditional_edges(
    "item_description", 
    fuzzy_conditional_node,
    {
        "store_fuzzy_result": "store_fuzzy_result",
        "category_mapping": "category_mapping"
    }
)

# store_fuzzy_result -> validate_fuzzy_itemname_with_llm (conditional)
# If validation TRUE -> fetch_item_info_node
# If validation FALSE -> category_mapping
graph.add_conditional_edges(
    "store_fuzzy_result",
    validate_fuzzy_itemname_with_llm,
    {
        "fetch_item_info_node": "fetch_item_info_node",
        "category_mapping": "category_mapping"
    }
)

# fetch_item_info_node -> END
graph.add_edge("fetch_item_info_node", END)

# category_mapping -> match_item_with_llm_node
graph.add_edge("category_mapping", "match_item_with_llm_node")

# match_item_with_llm_node -> END
graph.add_edge("match_item_with_llm_node", END)

app = graph.compile()


def process_ocr_items(ocr_result: dict, neo4j_conn: Optional[Neo4jConnection] = None) -> list[dict]:
    """
    Process line items from OCR result and map them to SAP item codes.
    Uses the LangGraph to orchestrate the flow.
    
    Args:
        ocr_result: The OCR result dictionary containing vendor_details and line_items
        neo4j_conn: Optional Neo4j connection (will create one if not provided)
    
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
    
    # Create Neo4j connection if not provided
    close_connection = False
    if neo4j_conn is None:
        try:
            neo4j_conn = Neo4jConnection()
            close_connection = True
        except Exception as e:
            logger.error(f"Failed to create Neo4j connection: {e}")
            neo4j_conn = None
    
    # Set the global Neo4j connection for graph nodes
    set_neo4j_connection(neo4j_conn)
    
    mapped_items = []
    
    try:
        for item in line_items:
            # Get item name from 'products' field (as per OCR processor schema)
            item_name = item.get("products", "") or item.get("description", "")
            
            if not item_name:
                logger.warning(f"Skipping item with no name: {item}")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing item: {item_name} from vendor: {vendor_name}")
            logger.info(f"{'='*60}")
            
            # Run the LangGraph to process the item
            # The graph handles the entire flow:
            # 1. item_description -> generate description
            # 2. fuzzy_conditional_node -> check fuzzy match
            #    - If match: store_fuzzy_result -> validate_fuzzy_itemname_with_llm
            #      - If valid: fetch_item_info_node -> END
            #      - If invalid: category_mapping -> match_item_with_llm_node -> END
            #    - If no match: category_mapping -> match_item_with_llm_node -> END
            
            result = app.invoke({
                "item_name": item_name,
                "vendor_name": vendor_name,
                "description": "",
                "category": "",
                "matched_item_code": "NO_MATCH",
                "matched_item_name": "",
                "fuzzy_item_code": "",
                "fuzzy_item_name": "",
                "fuzzy_validated": False,
                "match_method": ""
            })
            
            # Build the output item with all relevant info
            mapped_item = {
                "original_item": item,
                "item_name": item_name,
                "vendor_name": vendor_name,
                "description": result.get("description", ""),
                "category": result.get("category", ""),
                "matched_item_code": result.get("matched_item_code", "NO_MATCH"),
                "matched_item_name": result.get("matched_item_name", ""),
                "fuzzy_item_code": result.get("fuzzy_item_code", ""),
                "fuzzy_item_name": result.get("fuzzy_item_name", ""),
                "fuzzy_validated": result.get("fuzzy_validated", False),
                "match_method": result.get("match_method", "none")
            }
            
            mapped_items.append(mapped_item)
            
            logger.info(f"Result: {item_name} -> {mapped_item['matched_item_code']} ({mapped_item['match_method']})")
            
    finally:
        # Clear the global Neo4j connection
        set_neo4j_connection(None)
        
        if close_connection and neo4j_conn:
            neo4j_conn.close()
    
    return mapped_items


if __name__ == "__main__":
    # Example: Test with sample OCR output
    sample_ocr_result = {
        "vendor_details": {
            "name": "Laxmi Steel",
            "address": "Kathmandu, Nepal",
            "contact_number": "01-1234567",
            "email": "info@laxmisteel.com",
            "pan_number": "123456789"
        },
        "line_items": [
            {
                "products": ",HOX FULL INDUSTRIAL OXY. 7.00 CU. M",
                "quantity": "100",
                "rate": "85",
                "amount": "8500"
            },
            {
                "products": "HOX FULL NITROGEN GAS 7.00 CU. M",
                "quantity": "50",
                "rate": "450",
                "amount": "22500"
            }
        ]
    }
    
    print("="*80)
    print("TESTING ITEM NAME MAPPER WITH OCR OUTPUT")
    print("="*80)
    
    # Process the OCR items
    mapped_items = process_ocr_items(sample_ocr_result)
    
    for i, item in enumerate(mapped_items, 1):
        print(f"\n--- Item {i} ---")
        print(f"Original: {item['item_name']}")
        print(f"Vendor: {item['vendor_name']}")
        print(f"Description: {item['description'][:100]}..." if len(item.get('description', '')) > 100 else f"Description: {item.get('description', '')}")
        print(f"Category: {item['category']}")
        print(f"Fuzzy Match: {item.get('fuzzy_item_code', '')} - {item.get('fuzzy_item_name', '')}")
        print(f"Fuzzy Validated: {item.get('fuzzy_validated', False)}")
        print(f"Final Matched Code: {item['matched_item_code']}")
        print(f"Final Matched Name: {item.get('matched_item_name', '')}")
        print(f"Match Method: {item.get('match_method', 'none')}")

