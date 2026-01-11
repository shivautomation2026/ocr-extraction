"""
Script to export items from Neo4j database into separate CSV files by category.
Each category gets its own CSV file in the assets/neo4j_exports directory.
"""

import csv
import os
from neo4j import GraphDatabase

# Neo4j connection settings
NEO4J_URI = "neo4j+s://18356c3c.databases.neo4j.io"
NEO4J_AUTH = ("neo4j", "fzEU2Yw_F07zwDGWqXlg7ynCrXM8bMuaReiZ0oKHDkg")


def get_all_categories(driver) -> list[dict]:
    """Get all categories from Neo4j."""
    with driver.session(database="neo4j") as session:
        result = session.run("""
            MATCH (c:Category)
            RETURN c.series AS series, c.name AS name
            ORDER BY c.series
        """)
        return [{"series": r["series"], "name": r["name"]} for r in result]


def get_items_by_category(driver, category_name: str) -> list[dict]:
    """Get all items belonging to a specific category."""
    with driver.session(database="neo4j") as session:
        result = session.run("""
            MATCH (i:Item)-[:BELONGS_TO]->(c:Category {name: $category_name})
            RETURN i.code AS ItemCode, 
                   i.name AS ItemName, 
                   i.uom_group_entry AS UoMGroupEntry,
                   i.inventory_uom_entry AS InventoryUoMEntry
            ORDER BY i.code
        """, category_name=category_name)
        return [dict(record) for record in result]


def sanitize_filename(name: str) -> str:
    """Convert category name to a valid filename."""
    # Replace special characters and spaces with underscores
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '&']
    filename = name
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    filename = filename.replace(' ', '_')
    # Remove consecutive underscores
    while '__' in filename:
        filename = filename.replace('__', '_')
    return filename.strip('_').lower()


def save_category_to_csv(items: list[dict], category_name: str, output_dir: str) -> str:
    """Save items of a category to a CSV file."""
    if not items:
        return None
    
    filename = f"{sanitize_filename(category_name)}.csv"
    filepath = os.path.join(output_dir, filename)
    
    fieldnames = ['ItemCode', 'ItemName', 'UoMGroupEntry', 'InventoryUoMEntry']
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)
    
    return filepath


def export_all_categories(output_dir: str = None):
    """
    Export all items from Neo4j, grouped by category, into separate CSV files.
    
    Args:
        output_dir: Directory to save CSV files. Defaults to assets/neo4j_exports
    """
    # Set default output directory
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "assets", "neo4j_exports")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    
    with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        driver.verify_connectivity()
        print("Connected successfully!")
        
        # Get all categories
        categories = get_all_categories(driver)
        print(f"Found {len(categories)} categories.")
        
        exported_files = []
        total_items = 0
        
        for category in categories:
            category_name = category["name"]
            series = category["series"]
            
            # Get items for this category
            items = get_items_by_category(driver, category_name)
            item_count = len(items)
            
            if item_count > 0:
                # Save to CSV
                filepath = save_category_to_csv(items, category_name, output_dir)
                exported_files.append({
                    "category": category_name,
                    "series": series,
                    "item_count": item_count,
                    "file": filepath
                })
                total_items += item_count
                print(f"  ✓ {category_name} (Series {series}): {item_count} items -> {os.path.basename(filepath)}")
            else:
                print(f"  - {category_name} (Series {series}): No items (skipped)")
        
        print(f"\n=== Export Summary ===")
        print(f"Total categories with items: {len(exported_files)}")
        print(f"Total items exported: {total_items}")
        print(f"Output directory: {output_dir}")
        
        return exported_files


def main():
    """Main function to run the export."""
    print("\n=== Neo4j Category Export ===\n")
    export_all_categories()
    print("\nDone!")


if __name__ == "__main__":
    main()
