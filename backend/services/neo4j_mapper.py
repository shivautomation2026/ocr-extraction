"""
Neo4j-based item mapping service.
Uses category classification and graph search to find matching item codes.
Falls back to LLM for closest match when exact match is not found.
"""

import logging
from typing import Optional
from neo4j import GraphDatabase
from rapidfuzz import fuzz, process
from pydantic import BaseModel
from google import genai
# from ..core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j connection settings
NEO4J_URI = "neo4j+s"
NEO4J_AUTH = ("neo4j", "asdf")

# Category keywords for initial classification
CATEGORY_KEYWORDS = {
    "ELECTRICAL": ["switch", "relay", "contactor", "cable", "motor", "capacitor", "fuse", "mcb", "battery", "transformer", "panel", "circuit", "wiring", "sensor", "plc", "vfd"],
    "MECHANICALS": ["bearing", "bolt", "nut", "washer", "seal", "pump", "valve", "cylinder", "coupling", "gear", "shaft", "belt", "pulley", "spring", "gasket", "flange"],
    "CIVIL": ["concrete", "cement", "brick", "sand", "gravel", "pipe", "nipple", "fitting", "construction", "plumbing", "tile", "mixer"],
    "IT & ACCESSORIES": ["computer", "network", "printer", "monitor", "ups", "software", "keyboard", "mouse", "laptop", "server", "router", "switch"],
    "HEAVY EQUIPMENTS & AUTOMOBILES": ["excavator", "loader", "bulldozer", "crane", "forklift", "automobile", "vehicle", "tractor", "compressor", "generator"],
    "Auxiliary Raw Material": ["castable", "refractory", "nozzle", "tundish", "laddle", "mould", "asbestos", "insulation", "lining"],
    "RAW MATERIAL": ["scrap", "coal", "iron", "sponge", "billet", "wire rod", "ingot", "ore"],
    "FINISHED GOODS": ["tmt", "bar", "finished", "rod", "angle", "channel"],
    "HEALTH & SAFETY": ["safety", "helmet", "gloves", "goggles", "mask", "vest", "harness", "fire"],
    "FUEL LUBRICANTS AND GAS": ["fuel", "diesel", "petrol", "gas", "lubricant", "oil", "grease"],
    "PRINTING & STATIONARY": ["paper", "pen", "notebook", "printing", "stationery", "file", "folder"],
    "OTHER": [],
}


class CategoryMatch(BaseModel):
    """Model for LLM category classification response."""
    category_name: str
    confidence: float
    reasoning: str


class ItemMatch(BaseModel):
    """Model for LLM item matching response."""
    item_code: str
    item_name: str
    confidence: float
    reasoning: str


class Neo4jItemMapper:
    """
    Maps item descriptions to item codes using Neo4j graph database.
    
    Flow:
    1. Classify item into a category (using keywords + LLM)
    2. Search Neo4j for items in that category
    3. Use fuzzy matching to find best match
    4. If no good match, use LLM to find closest related item
    """
    
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        self.gemini_client = genai.Client(api_key="AIzaSyDh0odMOcaV4C6h17DByQR7GwDApJOHKfA")
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
    
    def _classify_by_keywords(self, item_description: str) -> Optional[str]:
        """
        Initial category classification using keyword matching.
        Returns category name or None if no match.
        """
        item_lower = item_description.lower()
        
        best_match = None
        best_score = 0
        
        for category, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in item_lower)
            if score > best_score:
                best_score = score
                best_match = category
        
        if best_score > 0:
            logger.info(f"Keyword classification: '{item_description}' -> {best_match} (score: {best_score})")
            return best_match
        
        return None
    
    def _classify_by_llm(self, item_description: str) -> Optional[str]:
        """
        Use LLM to classify item into a category.
        """
        categories = list(CATEGORY_KEYWORDS.keys())
        
        try:
            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=f"""
                    You are an industrial item classifier. Classify the following item into one of these categories:
                    {categories}
                    
                    Item description: "{item_description}"
                    
                    Analyze the item and determine which category it most likely belongs to.
                    Consider the item's function, material, and typical industrial usage.
                """,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': CategoryMatch
                }
            )
            
            result: CategoryMatch = response.parsed
            logger.info(f"LLM classification: '{item_description}' -> {result.category_name} (confidence: {result.confidence})")
            
            # Validate category exists
            if result.category_name in categories:
                return result.category_name
            
            # Try fuzzy match on category names
            best_cat = process.extractOne(result.category_name, categories, scorer=fuzz.ratio)
            if best_cat and best_cat[1] >= 70:
                return best_cat[0]
            
            return "OTHER"
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return None
    
    def classify_item_category(self, item_description: str) -> str:
        """
        Classify an item into a category.
        First tries keyword matching, falls back to LLM.
        """
        # Try keyword-based classification first
        category = self._classify_by_keywords(item_description)
        
        if category:
            return category
        
        # Fall back to LLM classification
        category = self._classify_by_llm(item_description)
        
        return category or "OTHER"
    
    def get_items_by_category(self, category_name: str) -> list[dict]:
        """
        Query Neo4j for all items in a specific category.
        """
        with self.driver.session(database="neo4j") as session:
            result = session.run("""
                MATCH (i:Item)-[:BELONGS_TO]->(c:Category {name: $category_name})
                RETURN i.code AS code, i.name AS name
                ORDER BY i.code
            """, category_name=category_name)
            
            items = [{"code": record["code"], "name": record["name"]} for record in result]
            logger.info(f"Found {len(items)} items in category '{category_name}'")
            return items
    
    def get_all_categories(self) -> list[str]:
        """Get all category names from Neo4j."""
        with self.driver.session(database="neo4j") as session:
            result = session.run("""
                MATCH (c:Category)
                RETURN c.name AS name
                ORDER BY c.name
            """)
            return [record["name"] for record in result]
    
    def fuzzy_match_item(self, item_description: str, category_items: list[dict], threshold: int = 70) -> Optional[dict]:
        """
        Find best matching item using fuzzy string matching.
        Returns item dict with code, name, and similarity score.
        """
        if not category_items:
            return None
        
        item_names = [item["name"].lower() for item in category_items]
        
        best_match = process.extractOne(
            item_description.lower(),
            item_names,
            scorer=fuzz.token_sort_ratio
        )
        
        if best_match and best_match[1] >= threshold:
            matched_name = best_match[0]
            similarity = best_match[1]
            
            # Find the corresponding item
            for item in category_items:
                if item["name"].lower() == matched_name:
                    logger.info(f"Fuzzy match: '{item_description}' -> {item['code']} ({similarity}% similarity)")
                    return {
                        "code": item["code"],
                        "name": item["name"],
                        "similarity": similarity,
                        "match_type": "fuzzy"
                    }
        
        return None
    
    def llm_find_closest_item(self, item_description: str, category_items: list[dict]) -> Optional[dict]:
        """
        Use LLM to find the closest matching item when fuzzy matching fails.
        """
        if not category_items:
            return None
        
        # Prepare items list for LLM
        items_text = "\n".join([f"- {item['code']}: {item['name']}" for item in category_items[:50]])  # Limit to 50 items
        
        try:
            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=f"""
                    You are an industrial item matching expert. Find the closest matching item from the list below.
                    
                    Item to match: "{item_description}"
                    
                    Available items:
                    {items_text}
                    
                    Analyze the item description and find the most similar item from the list.
                    Consider:
                    - Function and purpose
                    - Physical characteristics
                    - Material type
                    - Size/specifications
                    - Brand/make if mentioned
                    
                    Select the item that would most likely be the same or a suitable substitute.
                    If no reasonable match exists, set confidence to 0.
                """,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': ItemMatch
                }
            )
            
            result: ItemMatch = response.parsed
            
            if result.confidence >= 0.5:
                logger.info(f"LLM match: '{item_description}' -> {result.item_code} (confidence: {result.confidence})")
                return {
                    "code": result.item_code,
                    "name": result.item_name,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "match_type": "llm"
                }
            else:
                logger.warning(f"LLM could not find confident match for: '{item_description}'")
                return None
                
        except Exception as e:
            logger.error(f"LLM item matching failed: {e}")
            return None
    
    def search_across_categories(self, item_description: str) -> Optional[dict]:
        """
        Search for item across all categories when category-specific search fails.
        """
        all_categories = self.get_all_categories()
        all_items = []
        
        for category in all_categories:
            items = self.get_items_by_category(category)
            for item in items:
                item["category"] = category
            all_items.extend(items)
        
        logger.info(f"Searching across {len(all_items)} items in all categories")
        
        # Try fuzzy matching on all items
        result = self.fuzzy_match_item(item_description, all_items, threshold=75)
        if result:
            return result
        
        # Fall back to LLM with top candidates from each category
        return self.llm_find_closest_item(item_description, all_items[:100])
    
    def map_item_to_code(self, item_description: str) -> dict:
        """
        Main method to map an item description to an item code.
        
        Flow:
        1. Classify item into category
        2. Get items from that category in Neo4j
        3. Try fuzzy matching
        4. If no match, try LLM matching
        5. If still no match, search across all categories
        
        Returns dict with:
        - code: Item code (or None if not found)
        - name: Matched item name
        - category: Category name
        - match_type: 'fuzzy', 'llm', or 'not_found'
        - confidence/similarity: Match score
        """
        logger.info(f"Mapping item: '{item_description}'")
        
        # Step 1: Classify into category
        category = self.classify_item_category(item_description)
        logger.info(f"Classified into category: {category}")
        
        # Step 2: Get items from category
        category_items = self.get_items_by_category(category)
        
        if not category_items:
            logger.warning(f"No items found in category '{category}', searching all categories")
            result = self.search_across_categories(item_description)
            if result:
                result["category"] = category
                return result
            return {
                "code": None,
                "name": None,
                "category": category,
                "match_type": "not_found",
                "message": "No matching item found"
            }
        
        # Step 3: Try fuzzy matching
        fuzzy_result = self.fuzzy_match_item(item_description, category_items)
        if fuzzy_result:
            fuzzy_result["category"] = category
            return fuzzy_result
        
        # Step 4: Try LLM matching within category
        llm_result = self.llm_find_closest_item(item_description, category_items)
        if llm_result:
            llm_result["category"] = category
            return llm_result
        
        # Step 5: Search across all categories
        logger.info("No match in category, searching across all categories")
        cross_result = self.search_across_categories(item_description)
        if cross_result:
            cross_result["original_category"] = category
            return cross_result
        
        return {
            "code": None,
            "name": None,
            "category": category,
            "match_type": "not_found",
            "message": "No matching item found in any category"
        }
    
    def map_multiple_items(self, item_descriptions: list[str]) -> list[dict]:
        """
        Map multiple item descriptions to item codes.
        """
        results = []
        for desc in item_descriptions:
            result = self.map_item_to_code(desc)
            result["input_description"] = desc
            results.append(result)
        return results


# Convenience function for quick mapping
def map_item(item_description: str) -> dict:
    """
    Quick function to map a single item description to an item code.
    Creates and closes Neo4j connection per call.
    """
    mapper = Neo4jItemMapper()
    try:
        return mapper.map_item_to_code(item_description)
    finally:
        mapper.close()


if __name__ == "__main__":
    # Test the mapper
    mapper = Neo4jItemMapper()
    
    test_items = [
        "3 phase contactor 25A",
        "M12 hex bolt stainless steel",
        "Hydraulic oil filter",
        "Ready mix concrete M25",
        "Safety helmet yellow",
    ]
    
    print("\n=== Testing Neo4j Item Mapper ===\n")
    
    for item in test_items:
        print(f"Input: {item}")
        result = mapper.map_item_to_code(item)
        print(f"Result: {result}")
        print("-" * 50)
    
    mapper.close()
