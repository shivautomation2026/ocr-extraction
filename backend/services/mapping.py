import json
import pandas as pd
from pandas.errors import EmptyDataError
from backend.database import collection
from fastapi import HTTPException
import logging
from pydantic import BaseModel
from rapidfuzz import fuzz, process
from ..core.config import settings
from google import genai
import re
from backend.services.sap_api import SAPClient
from backend.services.neo4j_mapper import Neo4jItemMapper
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Item(BaseModel):
    Series: int
    UoMGroupEntry: int
    ItemName: str

class AccountCode(BaseModel):
    ItemName: str
    AccountCode: str
    

class Mapper:
    def __init__(self):
        self.sap_client = SAPClient()
        self.gemini_client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.neo4j_mapper = Neo4jItemMapper()  # Initialize Neo4j mapper
        self.sap_client.save_items_to_csv()
        self.sap_client.save_item_groups_to_csv()
        self.sap_client.save_business_partners()
        self.sap_client.save_uom_groups_to_csv()
        
        try:
            self.vendor_names_with_codes = pd.read_csv('backend/assets/vendor_list.csv')
        except (FileNotFoundError, EmptyDataError):
            logger.error("Warning: 'vendor_list.csv' not found or empty. Vendor code validation will be disabled.")
            self.vendor_names_with_codes = None

        try:
            self.item_list_df = pd.read_csv('backend/assets/item_list.csv')
            self.item_names_list = self.item_list_df["ItemName"].str.lower().dropna().to_list()
            logger.info("Successfully loaded item_list.csv for product mapping.")
        except (FileNotFoundError, EmptyDataError):
            logger.error("Warning: 'item_list.csv' not found or empty. Item code mapping will be disabled.")
            self.item_list_df = None
            self.item_names_list = None

        try:
            item_groups_df = pd.read_csv('backend/assets/item_groups.csv')
            self.item_groups_string = item_groups_df.to_dict(orient='records')
            logger.info("Successfully loaded item_groups.csv for product group mapping.")
        except (FileNotFoundError, EmptyDataError):
            logger.error("Warning: 'item_groups.csv' not found or empty. Item group mapping will be disabled.")
            self.item_groups_string = None

        try:
            uom_groups_df = pd.read_csv('backend/assets/uom_groups.csv')
            self.uom_groups_string = uom_groups_df.to_dict(orient='records')
            logger.info("Successfully loaded uom_groups.csv for UoM group mapping.")
        except (FileNotFoundError, EmptyDataError):
            logger.error("Warning: 'uom_groups.csv' not found or empty. UoM group mapping will be disabled.")
            self.uom_groups_string = None

        try:
            with open('backend/assets/costing_codes.json') as f:
                self.costing_code = json.load(f)
        except Exception as e:
            logger.error(f"Error loading costing_codes.json: {e}")

        try:
            account_codes_df = pd.read_csv('backend/assets/account_codes.csv')
            self.account_codes_string = account_codes_df.to_dict(orient='records')
            logger.info("Successfully loaded account_codes.csv for account code mapping.")
        except (FileNotFoundError, EmptyDataError):
            logger.error("Warning: 'account_codes.csv' not found or empty. Account code mapping will be disabled.")
            self.account_codes_string = None


    def find_similar_vendor(self, document_uid: int, threshold: int = 80):
        try:
            self.sap_client.save_business_partners()
            
            document_json = collection.find_one({"uid": document_uid}, {"_id": 0, "extracted_details": 1})
            if not document_json:
                raise HTTPException(status_code=404, detail=f"Document with UID {document_uid} not found.")
            incoming_json_for_code = document_json.get("extracted_details", {})
            if not incoming_json_for_code:
                raise HTTPException(status_code=404, detail="No 'extracted_details' found in the document.")

            incoming_vendor_name = incoming_json_for_code['vendor_details']['name']

            vendor_names_list = self.vendor_names_with_codes["CardName"].str.lower().dropna().tolist()
            # vendor_names = [name[0] for name in vendor_names_list]
            logger.info("Starting vendor name matching process.")

            best_match = process.extractOne(
                incoming_vendor_name.lower(), 
                vendor_names_list, 
                scorer=fuzz.ratio
            )
            logger.info(best_match)
            
            if best_match and best_match[1] >= threshold:
                matched_vendor_name = best_match[0]
                similarity_score = best_match[1]

                vendor_row = self.vendor_names_with_codes[self.vendor_names_with_codes["CardName"].str.lower() == matched_vendor_name]
                if not vendor_row.empty:
                    vendor_code = vendor_row.iloc[0]['CardCode']
                    collection.update_one(
                        {"uid": document_uid},
                        {"$set": {"extracted_details.vendor_details.code": vendor_code}}
                    )
                    logger.info(f"Updated vendor code for document UID {document_uid} to '{vendor_code}' "
                            f"(matched '{incoming_vendor_name}' to '{matched_vendor_name}' with {similarity_score}% similarity).")
            else:
                logger.warning(f"No similar vendor name found for '{incoming_vendor_name}' with sufficient similarity.")
        except Exception as e:

            logger.error(f"Error in find_similar_vendor: {e}")

    def map_items_to_codes(self, document_uid: int):
        self.sap_client.save_item_groups_to_csv()

        if self.item_list_df is None:
            logger.warning("Item list CSV not loaded. Skipping item code mapping.")
            return
        
        try:
            document_json = collection.find_one({"uid": document_uid}, {"_id": 0, "extracted_details": 1})
            if not document_json:
                logger.error(f"Document with UID {document_uid} not found.")
                return
            
            incoming_json = document_json.get("extracted_details", {})
            if not incoming_json:
                logger.error("No 'extracted_details' found in the document.")
                return
            
            line_items = incoming_json.get('line_items', [])
            if not line_items:
                logger.warning("No line_items found in document.")
                return
            
            for id, item in enumerate(line_items):
                item_code = item.get('ItemCode')
                item_desc = item.get('products')
                if not item_desc:
                    logger.warning(f"No product description found for line item {id}")
                    continue

                if not item_code:
                    # First, try Neo4j-based mapping (category-aware search)
                    logger.info(f"Attempting Neo4j-based mapping for: '{item_desc}'")
                    neo4j_result = self.neo4j_mapper.map_item_to_code(item_desc)
                    
                    if neo4j_result.get("code"):
                        item_code = neo4j_result["code"]
                        match_type = neo4j_result.get("match_type", "unknown")
                        category = neo4j_result.get("category", "unknown")
                        
                        # Get UoM entry from item list
                        item_row = self.item_list_df[self.item_list_df["ItemCode"] == item_code]
                        if not item_row.empty:
                            uom_entry = int(item_row.iloc[0]['InventoryUoMEntry'])
                        else:
                            uom_entry = None
                        
                        collection.update_one(
                            {"uid": document_uid},
                            {"$set": {
                                f"extracted_details.line_items.{id}.ItemCode": item_code,
                                f"extracted_details.line_items.{id}.UoMCode": uom_entry,
                                f"extracted_details.line_items.{id}.MatchType": match_type,
                                f"extracted_details.line_items.{id}.Category": category
                            }}
                        )
                        logger.info(f"Neo4j mapped line item {id}: '{item_desc}' -> {item_code} "
                                  f"(category: {category}, match: {match_type})")
                        continue
                    
                    # Fallback to original fuzzy matching on full item list
                    logger.info(f"Neo4j mapping failed, falling back to fuzzy matching for: '{item_desc}'")
                    best_match = process.extractOne(
                        item_desc.lower(), 
                        self.item_names_list, 
                        scorer=fuzz.ratio
                    )
                    logger.info(f"Best item match: {best_match}")

                    if best_match and best_match[1] >= 80:
                        matched_item_name = best_match[0]
                        similarity_score = best_match[1]

                        item_row = self.item_list_df[self.item_list_df["ItemName"].str.lower() == matched_item_name]
                        if not item_row.empty:
                            item_code = item_row.iloc[0]['ItemCode']
                            uom_entry = int(item_row.iloc[0]['InventoryUoMEntry']) # Convert numpy.int64 to python int
                            
                            collection.update_one(
                                {"uid": document_uid},
                                {"$set": {f"extracted_details.line_items.{id}.ItemCode": item_code, f"extracted_details.line_items.{id}.UoMCode": uom_entry}}
                            )
                            logger.info(f"Mapped line item {id}: '{item_desc}' to ItemCode '{item_code}' "
                                    f"(matched to '{matched_item_name}' with {similarity_score}% similarity).")
                    else:
                        logger.warning(f"No matching ItemCode found for line item {id}: '{item_desc}'")
                        time.sleep(5) # Add a delay before creating a new item
                        new_item_codes = self.create_new_items(item_desc)
                        time.sleep(5) # Add a delay between API calls
                        account_code_data = self.map_account_codes(item_desc)
                        if new_item_codes and account_code_data:
                            collection.update_one(
                                {"uid": document_uid},
                                {"$set": {f"extracted_details.line_items.{id}.ItemCode": new_item_codes['ItemCode'], f"extracted_details.line_items.{id}.UoMCode": new_item_codes['InventoryUoMEntry'], f"extracted_details.line_items.{id}.AccountCode": account_code_data['AccountCode']}}
                            )
                            logger.info(f"Created and mapped new item for line item {id}: '{item_desc}' with ItemCode '{new_item_codes['ItemCode']}'")
                        else:
                            logger.error(f"Failed to create or map account code for new item: '{item_desc}'")

                time.sleep(1) # Add a 1-second delay to avoid hitting rate limits

        except Exception as e:
            logger.error(f"Error in map_items_to_codes: {e}")
            
    def create_new_items(self, item_description: str):
        try:
            logger.info(f"No item found. Creating new item for description: {item_description}")
            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=f"""
                    You are given:
                    1. An item description: "{item_description}"
                    2. A reference list for item groups (used to determine Series):
                    {self.item_groups_string}
                    3. A reference list for UoM groups (used to determine UoMGroupEntry from key: Code):
                    {self.uom_groups_string}

                    Your task:
                    1. Carefully analyze the item description and infer which category or group (from the item groups list) it most likely belongs to.
                    - Match the item semantically or contextually to the most relevant GroupName.
                    - From that record, extract and return its corresponding Series.
                    2. Similarly, identify which UoM type (from the UoM groups list) best fits the item.
                    - Match the item meaning to the most appropriate Code.
                    - From that record, extract and return its corresponding UoMGroupEntry.
                    3. If no confident match is found, set the respective field (Series or UoMGroupEntry) to null.
                    
                    Be factual and concise. Do not invent or assume information beyond the given references.

                """,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': list[Item]
                }
            )
            new_item : list[Item] = response.parsed
            for item in new_item:
                item_data = {
                    "ItemName": item.ItemName,
                    "Series": item.Series,
                    "UoMGroupEntry": item.UoMGroupEntry
                }
    
                return self.sap_client.post_items_to_sap(item_data)

        except Exception as e:
            logger.error(f"Error creating new item: {e}")
            return None

    def map_costing_code(self, item_name: str):
        try:
            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=f"""I have a list of cost_center_code, const_center_name, account_code, account_name: {json.dumps(self.costing_code)}. Please map the item name '{item_name}' to the appropriate cost_center_code and account code from the provided cost_center_name and account_name. Make references close as much as you can. Generate a JSON object with the item name and the corresponding cost_center_code and account_code. Give me in a plain json object without any markdown format. Do not hallucinate.
                Example:
                {
                    "ItemName": "Sample Item",
                    "CostingCode": "...",
                    "AccountCode": "..."
                }
                """,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': list[AccountCode]
                })
            mapped_data = response.text
            # match = re.search(r'```json\s*(\{.*?\})\s*```', mapped_data, re.DOTALL)
            # if match:
            #     matched_json = match.group(1)
            #     cost_codes = json.loads(matched_json)
            #     return cost_codes      
                
            
        except Exception as e:
            logger.error(f"Error mapping costing code for item {item_name}: {e}")
            
    def map_account_codes(self, item_name:str):
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=f"""
                    You are given a list of account records containing 'AccountCode' and 'Name' fields:
                    {self.account_codes_string}

                    Your task:
                    1. Analyze the provided list and identify which account category best matches the given item: "{item_name}".
                    2. The item name may not exactly match any Name in the list, but it likely belongs to one of the categories.
                    - For example, an item like "Speaker" or "Mouse" should fall under categories related to IT, Office Equipment, or Administrative Expenses.
                    3. Select the AccountCode whose Name best represents the correct accounting category for the given item.
                    4. If no relevant category can be found, return "AccountCode": null.
                    Do not include any explanations, notes, or extra text outside the JSON.
                """
            ,config={
                    'response_mime_type': 'application/json',
                    'response_schema': AccountCode
        })
            mapped_data = response.text
            
            if mapped_data:
                return json.loads(mapped_data)
            return None
        
        except Exception as e:
            logger.error(f"Error mapping account code for item {item_name}: {e}")
            return None
