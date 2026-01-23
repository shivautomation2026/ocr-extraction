"""
Script to add items and categories to Neo4j database.
Items are classified into categories based on their item code prefix
and item name patterns.
"""

import csv
import os
from neo4j import GraphDatabase
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_AUTH = (os.getenv("NEO4J_USERNAME", ""), os.getenv("NEO4J_PASSWORD", ""))

# Mapping of item code prefixes to category Series (aligned with series.csv)
PREFIX_TO_CATEGORY = {
    "EL": 105,   # Electrical
    "ME": 107,   # Mechanicals
    "HE": 108,   # Heavy Equipments & Automobiles
    "CV": 110,   # Civil
    "MS": 111,   # Mesh
    "AR": 112,   # Auxiliary Raw Material
    "HS": 113,   # Health & Safety
    "FG": 114,   # Finished Goods
    "FL": 115,   # Fuel Lubricant and gas
    "PS": 116,   # Printing & Stationary
    "TR": 117,   # Trading
    "RM": 118,   # Raw material
    "IT": 119,   # IT & Accessories
    "BP": 120,   # Business Promotion & Marketing
    "RR": 121,   # Wastages
    "OT": 109,   # Other
    "AS": 109,   # Other (Asset items -> Other)
    "SV": 143,   # Service
}


def load_series_categories(filepath: str) -> dict:
    """Load series categories from CSV and return as dict {Series: SeriesName}."""
    categories = {}
    with open(filepath, "r", encoding="utf-8") as f:
        # Read the file and handle tab-separated values
        lines = f.readlines()
        if not lines:
            return categories
        
        # Skip header and parse data rows
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            # Split by comma or tab
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2:
                try:
                    series_num = int(parts[0])
                    series_name = parts[1].strip()
                    if series_name:
                        categories[series_num] = series_name
                except (ValueError, IndexError):
                    continue
    
    return categories


def load_items(filepath: str) -> list[dict]:
    """Load items from CSV."""
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append({
                "item_code": row["ItemCode"],
                "item_name": row["ItemName"],
                "uom_group_entry": row.get("UoMGroupEntry", ""),
                "inventory_uom_entry": row.get("InventoryUoMEntry", ""),
            })
    return items


def classify_item(item_code: str, item_name: str) -> Optional[int]:
    """
    Classify an item based on its code prefix.
    Returns the category Series number.
    """
    # Extract prefix (first 2 letters)
    prefix = "".join(c for c in item_code[:2] if c.isalpha()).upper()
    
    if prefix in PREFIX_TO_CATEGORY:
        return PREFIX_TO_CATEGORY[prefix]
    
    # Fallback: try to classify by item name keywords
    item_name_upper = item_name.upper()
    
    keyword_mapping = {
        105: ["ELECTRICAL", "SWITCH", "RELAY", "CONTACTOR", "CABLE", "MOTOR", "CAPACITOR", "FUSE", "MCB", "BATTERY"],
        107: ["MECHANICAL", "BEARING", "BOLT", "NUT", "WASHER", "SEAL", "PUMP", "VALVE", "CYLINDER", "COUPLING"],
        110: ["CIVIL", "CONCRETE", "CEMENT", "BRICK", "SAND", "GRAVEL", "PIPE", "NIPPLE"],
        119: ["IT", "COMPUTER", "NETWORK", "PRINTER", "MONITOR", "UPS", "SOFTWARE"],
        108: ["EXCAVATOR", "LOADER", "BULLDOZER", "CRANE", "FORKLIFT", "AUTOMOBILE", "VEHICLE"],
        112: ["CASTABLE", "REFRACTORY", "NOZZLE", "TUNDISH", "LADDLE", "MOULD"],
        118: ["SCRAP", "COAL", "IRON", "SPONGE", "BILLET", "WIRE ROD"],
        114: ["TMT", "BAR", "FINISHED"],
        113: ["SAFETY", "HELMET", "GLOVES", "GOGGLES", "MASK"],
        115: ["FUEL", "DIESEL", "PETROL", "GAS", "LUBRICANT", "OIL"],
    }
    
    for category_series, keywords in keyword_mapping.items():
        if any(kw in item_name_upper for kw in keywords):
            return category_series
    
    # Default to OTHER if no match
    return 109


def create_neo4j_schema(driver):
    """Create constraints and indexes for better performance."""
    with driver.session(database="neo4j") as session:
        # Create constraints for unique identifiers
        session.run("""
            CREATE CONSTRAINT category_series IF NOT EXISTS
            FOR (c:Category) REQUIRE c.series IS UNIQUE
        """)
        session.run("""
            CREATE CONSTRAINT item_code IF NOT EXISTS
            FOR (i:Item) REQUIRE i.code IS UNIQUE
        """)
        print("Schema constraints created.")


def add_categories_to_neo4j(driver, groups: dict):
    """Add all categories to Neo4j."""
    with driver.session(database="neo4j") as session:
        for series, group_name in groups.items():
            session.run("""
                MERGE (c:Category {series: $series})
                SET c.name = $name
            """, series=series, name=group_name)
        print(f"Added {len(groups)} categories to Neo4j.")


def add_items_to_neo4j(driver, items: list[dict], groups: dict):
    """Add items to Neo4j and create relationships to categories."""
    with driver.session(database="neo4j") as session:
        added_count = 0
        for item in items:
            category_series = classify_item(item["item_code"], item["item_name"])
            
            if category_series and category_series in groups:
                session.run("""
                    MERGE (i:Item {name: $name})
                    SET i.uom_group_entry = $uom_group,
                        i.inventory_uom_entry = $inv_uom
                    WITH i
                    OPTIONAL MATCH (existing:Item {code: $code})
                    WITH i, existing
                    FOREACH (_ IN CASE WHEN existing IS NULL OR existing = i THEN [1] ELSE [] END |
                        SET i.code = $code
                    )
                    WITH i
                    MATCH (c:Category {series: $category_series})
                    MERGE (i)-[:BELONGS_TO]->(c)
                """, 
                    code=item["item_code"],
                    name=item["item_name"],
                    uom_group=item["uom_group_entry"],
                    inv_uom=item["inventory_uom_entry"],
                    category_series=category_series
                )
                added_count += 1
        
        print(f"Added {added_count} items to Neo4j with category relationships.")


def get_items_by_category(driver, category_name: str) -> list[dict]:
    """Query items by category name."""
    with driver.session(database="neo4j") as session:
        result = session.run("""
            MATCH (i:Item)-[:BELONGS_TO]->(c:Category {name: $category_name})
            RETURN i.code AS code, i.name AS name
            ORDER BY i.code
        """, category_name=category_name)
        return [{"code": record["code"], "name": record["name"]} for record in result]


def get_category_statistics(driver) -> list[dict]:
    """Get count of items per category."""
    with driver.session(database="neo4j") as session:
        result = session.run("""
            MATCH (c:Category)
            OPTIONAL MATCH (i:Item)-[:BELONGS_TO]->(c)
            RETURN c.name AS category, c.series AS series, COUNT(i) AS item_count
            ORDER BY item_count DESC
        """)
        return [{"category": r["category"], "series": r["series"], "count": r["item_count"]} 
                for r in result]


def main():
    """Main function to load data and populate Neo4j."""
    # Get the assets directory path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(base_dir, "assets")
    
    series_path = os.path.join(assets_dir, "series.csv")
    item_list_path = os.path.join(assets_dir, "item_list.csv")
    
    # Load data from CSVs
    print("Loading data from CSV files...")
    categories = load_series_categories(series_path)
    items = load_items(item_list_path)
    
    print(f"Loaded {len(categories)} categories from series.csv and {len(items)} items.")
    print(f"Categories: {dict(list(categories.items())[:5])}... (showing first 5)")
    
    # Get Neo4j connection details from environment or script
    neo4j_uri = NEO4J_URI or os.getenv("NEO4J_URI", "")
    neo4j_user = os.getenv("NEO4J_USER", "")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "")
    neo4j_auth = NEO4J_AUTH
    
    if (not neo4j_auth[0] or not neo4j_auth[1]) and neo4j_user and neo4j_password:
        neo4j_auth = (neo4j_user, neo4j_password)
    
    if not neo4j_uri:
        raise RuntimeError(
            "Neo4j URI is not set. Set NEO4J_URI in the script or export NEO4J_URI env var."
        )
    if not neo4j_auth[0] or not neo4j_auth[1]:
        raise RuntimeError(
            "Neo4j auth is not set. Set NEO4J_AUTH in the script or export NEO4J_USER/NEO4J_PASSWORD."
        )
    
    # Connect to Neo4j and populate
    print(f"Connecting to Neo4j at {neo4j_uri}...")
    
    with GraphDatabase.driver(neo4j_uri, auth=neo4j_auth) as driver:
        driver.verify_connectivity()
        print("Connected successfully!")
        
        # Create schema
        create_neo4j_schema(driver)
        
        # Step 1: Add all categories first
        print("\nStep 1: Creating categories...")
        add_categories_to_neo4j(driver, categories)
        
        # Step 2: Add items with BELONGS_TO relationships
        print("\nStep 2: Adding items with BELONGS_TO relationships...")
        add_items_to_neo4j(driver, items, categories)
        
        # Print statistics
        print("\n--- Category Statistics ---")
        stats = get_category_statistics(driver)
        for stat in stats:
            print(f"  {stat['category']}: {stat['count']} items")
        
        print("\nDone!")


if __name__ == "__main__":
    main()