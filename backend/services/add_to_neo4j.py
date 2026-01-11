"""
Script to add items and categories to Neo4j database.
Items are classified into categories based on their item code prefix
and item name patterns.
"""

import csv
import os
from neo4j import GraphDatabase
from typing import Optional

# Neo4j connection settings
NEO4J_URI = ""
NEO4J_AUTH = ("", "")

# Mapping of item code prefixes to category Series
PREFIX_TO_CATEGORY = {
    "EL": 103,   # ELECTRICAL
    "ME": 104,   # MECHANICALS
    "CV": 107,   # CIVIL
    "IT": 108,   # IT & ACCESSORIES
    "HE": 106,   # HEAVY EQUIPMENTS & AUTOMOBILES
    "BP": 112,   # BUSINESS PROMOTION & MARKETING
    "FG": 113,   # FINISHED GOODS
    "RM": 114,   # RAW MATERIAL
    "AR": 105,   # Auxiliary Raw Material
    "PS": 109,   # PRINTING & STATIONARY
    "OT": 111,   # OTHER
    "AS": 101,   # Asset
    "HS": 115,   # HEALTH & SAFETY
    "FL": 110,   # FUEL LUBRICANTS AND GAS
}


def load_item_groups(filepath: str) -> dict:
    """Load item groups from CSV and return as dict {Series: GroupName}."""
    groups = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            series = int(row["Series"])
            groups[series] = row["GroupName"]
    return groups


def load_items(filepath: str) -> list[dict]:
    """Load items from CSV."""
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip items with invalid UoM entries
            if row.get("UoMGroupEntry") == "-1":
                continue
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
        103: ["ELECTRICAL", "SWITCH", "RELAY", "CONTACTOR", "CABLE", "MOTOR", "CAPACITOR", "FUSE", "MCB", "BATTERY"],
        104: ["MECHANICAL", "BEARING", "BOLT", "NUT", "WASHER", "SEAL", "PUMP", "VALVE", "CYLINDER", "COUPLING"],
        107: ["CIVIL", "CONCRETE", "CEMENT", "BRICK", "SAND", "GRAVEL", "PIPE", "NIPPLE"],
        108: ["IT", "COMPUTER", "NETWORK", "PRINTER", "MONITOR", "UPS", "SOFTWARE"],
        106: ["EXCAVATOR", "LOADER", "BULLDOZER", "CRANE", "FORKLIFT", "AUTOMOBILE", "VEHICLE"],
        105: ["CASTABLE", "REFRACTORY", "NOZZLE", "TUNDISH", "LADDLE", "MOULD"],
        114: ["SCRAP", "COAL", "IRON", "SPONGE", "BILLET", "WIRE ROD"],
        113: ["TMT", "BAR", "FINISHED"],
        115: ["SAFETY", "HELMET", "GLOVES", "GOGGLES", "MASK"],
        110: ["FUEL", "DIESEL", "PETROL", "GAS", "LUBRICANT", "OIL"],
    }
    
    for category_series, keywords in keyword_mapping.items():
        if any(kw in item_name_upper for kw in keywords):
            return category_series
    
    # Default to OTHER if no match
    return 111


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
                    MERGE (i:Item {code: $code})
                    SET i.name = $name,
                        i.uom_group_entry = $uom_group,
                        i.inventory_uom_entry = $inv_uom
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
    
    item_groups_path = os.path.join(assets_dir, "item_groups.csv")
    item_list_path = os.path.join(assets_dir, "item_list.csv")
    
    # Load data from CSVs
    print("Loading data from CSV files...")
    groups = load_item_groups(item_groups_path)
    items = load_items(item_list_path)
    
    print(f"Loaded {len(groups)} categories and {len(items)} items.")
    
    # Connect to Neo4j and populate
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    
    with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        driver.verify_connectivity()
        print("Connected successfully!")
        
        # Create schema
        create_neo4j_schema(driver)
        
        # Add categories
        add_categories_to_neo4j(driver, groups)
        
        # Add items with relationships
        add_items_to_neo4j(driver, items, groups)
        
        # Print statistics
        print("\n--- Category Statistics ---")
        stats = get_category_statistics(driver)
        for stat in stats:
            print(f"  {stat['category']}: {stat['count']} items")
        
        print("\nDone!")


if __name__ == "__main__":
    main()
