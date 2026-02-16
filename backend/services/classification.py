from ..core.config import settings
import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from ..database import collection
from typing import Dict, Optional
import re
import requests
import pandas as pd
import logging

# Load environment variables from .env file
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

total_uploads = collection.count_documents({"file_name": {"$exists":True}})

class Classifier:
    def __init__(self):
        api_key = settings.GOOGLE_API_KEY
        if not api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables")
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        self.client = ChatGoogleGenerativeAI(model=settings.GEMINI_MODEL_NAME)
        logger.info("Gemini model initialized.")
    
    def parse_invoice_json(self, raw_json_string: str) -> Optional[Dict]:
        """Parses a raw string to extract and load a JSON object."""

        json_start = raw_json_string.find('{')
        json_end = raw_json_string.rfind('}')

        if json_start != -1 and json_end != -1:
            json_substring = raw_json_string[json_start : json_end + 1]
            try:
                invoice_data_dict = json.loads(json_substring)
                return invoice_data_dict
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON: {e}")
                return None
        else:
            logger.error("Could not find a valid JSON object in the string.")
            return None
    
    def classify_invoice(self, invoice_json: dict, model) -> str:
        """
        Classifies an invoice based on its content using a Gemini model.
        Also checks for a specific net_amount condition.
        """
        try:
            total_amount = invoice_json.get("payment_details", {}).get("grand_total")

            if total_amount is not None:
                try:
                    cleaned_amount_str = re.sub(r'[^0-9.]', '', str(total_amount))
                    total_amount_float = float(cleaned_amount_str)

                    if total_amount_float < 2000:
                        logger.info("Identified as outgoing_payment based on grand_total < 2000")
                        return "outgoing_payment"
                except (ValueError, TypeError) as num_err:
                        logger.error(f"Could not convert grand_total to number: {total_amount} - {num_err}")
                        # Continue to model classification if number conversion fails
        except Exception as e:
            # Catch any unexpected errors during the initial check
            logger.error(f"Error during grand_total check: {e}")
        # If the net_amount condition is not met, use the model for classification
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a document classifier for SAP systems. Based on the invoice JSON data, classify the document into one of the following types:
    - ap_invoice
    - ap_invoice_with_lc

    Your classification should depend on the nature of the document:
    - 'ap_invoice' is a standard vendor invoice
    - 'ap_invoice_with_lc' includes reference to LC (Letter of Credit) in payment mode, particulars, or vendor behavior (if invoice_details.lc_no is preset it is ap_invoice_with_lc)

    Respond with only one of the labels: ap_invoice or ap_invoice_with_lc.
    """),
            ("user", "Here is the invoice data:\n{invoice_json}")
        ])

        chain = prompt | model

        try:
            response = chain.invoke({
                "invoice_json": json.dumps(invoice_json, indent=2)
            })
            return response.content.strip()
        except Exception as e:
            logger.error(f"Classification failed: {type(e).__name__} - {e}")
            return f"Classification failed: {type(e).__name__} - {e}"

    def process_classification(self, document_id: int):
        """Fetches document by ID and processes classification based on extracted details."""
        logger.info("Attempting to process classification for document ID: {document_id}")
        try:
            # Fetch the document from MongoDB using the uid
            document = collection.find_one({"uid": document_id}, {"extracted_details": 1, "_id": 0})

            if not document:
                logger.error(f"Document with ID {document_id} not found in database.")
                return "Document not found."

            # Directly access the extracted_details field from the document dictionary
            extracted_details = document.get("extracted_details")

            if not extracted_details:
                logger.error(f"No 'extracted_details' field found for document ID: {document_id}")
                return "No extracted details found for this document."

            # Check if extracted_details is a string (as it sometimes might be stored)
            # If it is a string, try to parse it as JSON
            if isinstance(extracted_details, str):
                try:
                    invoice_data_dict = json.loads(extracted_details)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from 'extracted_details' for document ID {document_id}: {e}")
                    return f"Error processing extracted details: Invalid JSON format ({e})"
            else:
                # Assume it's already the correct dictionary format
                invoice_data_dict = extracted_details
                logger.info(f"'extracted_details' for document ID {document_id} is already a dictionary.")

            # Ensure we have a dictionary before passing to classify_invoice
            if not isinstance(invoice_data_dict, dict):
                logger.info(f"'extracted_details' for document ID {document_id} is not in a valid dictionary format after processing.")
                return "Extracted details are not in a valid dictionary format for classification."

            # Get the Gemini model
            model = self.client

            # Classify the invoice
            classification_result = self.classify_invoice(invoice_data_dict, model)

            if classification_result == 'ap_invoice':
                gl_classified = self.gl_account_classifier(document_id)
                collection.update_one({"uid": document_id}, {"$set":  {"gl_classification": gl_classified}})

            else: 
                logger.info("Not a ap_invoice")

            return classification_result

        except Exception as e:
            logger.error(f"An unexpected error occurred during classification for document ID {document_id}: {e}")
            return f"An internal error occurred: {str(e)}"

    def match_vendor_name(self, document_id: int):
        try:
            API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/sentence-similarity"
            access_token = os.getenv("HF_API_KEY")
            headers = {
                "Authorization": f"Bearer {access_token}",
            }
            df = pd.read_csv("backend/assets/vendor_list.csv")
            logger.info("Loaded vendor names from CSV.")
            vendor_names = df.values.tolist()

            def query(payload):
                response = requests.post(API_URL, headers=headers, json=payload)
                return response.json()
            
            document = collection.find_one({"uid": document_id}, {"extracted_details": 1, "_id": 0})
            vendor_name = document.get("extracted_details", {}).get("vendor_details", {}).get("name")
            vendor_name_list = [list[0] for list in vendor_names]

            output = query({
                "inputs": {
                    "source_sentence": vendor_name,
                    "sentences": vendor_name_list
                }
            })

            if output and isinstance(output, list):
                scores = output[0] if isinstance(output[0], list) else output
                
                if not isinstance(scores, list):
                    logger.error(f"Unexpected scores format: {type(scores)}")
                    return 

                highest_score = max(scores)
                best_index = scores.index(highest_score)
                
                best_match = vendor_name_list[best_index]
                            
                logger.info(f"Best match found while classifying: {best_match}")
                logger.info(f"Similarity score: {highest_score:.2%}")
                logger.info(f"Index in list: {best_index}")

                collection.update_one(
                    {"uid": document_id},
                    {"$set": {"extracted_details.vendor_details.name": best_match}}
                )
                logger.info(f"Vendor name updated in database for document ID {document_id} to {best_match}.")
            else:
                logger.error(f"Invalid API response format: {type(output)}")
                logger.error(f"Response content: {output}")

        except Exception as e:
            logger.error(f"An unexpected error occurred during vendor name matching for document ID {document_id}: {e}")
            return f"An internal error occurred: {str(e)}"

    def gl_account_classifier(self, document_id:id):
        document = collection.find_one({"uid": document_id}, {"extracted_details": 1, "_id": 0})
        raw_vendor_extracted_details = document.get("extracted_details", {})
        vendor_extracted_details = json.dumps(raw_vendor_extracted_details)
        
        invoice_json_data = vendor_extracted_details
        gl_mapping_json_data = """
            {
            "Advertisement Expenses": [
                "Sales-KTV ( 5 sec Headline break all news Sarbottam steelTVC cost of KrV dated Magh 1-30 2081 ) (As per RO)",
                "Toward the Cost of Facebook Page Management",
                "Advertisement tax and service",
                "Radio Advertising",
                "Volume Branding"
            ],
            "CARGO FEE": [
                "Consignment Note",
                "DHL Express or any cargo company",
                "Cargo and Courier"
            ],
            "Cleaning Expenses": [
                "Harpic Dettol Lizol Exo Odonil (bhatbhateni)"
            ],
            "Electricity Expenses": [
                "related to energy companies (electricit charges of a certain month in line item)"
            ],
            "FURNITURE & FIXTURE": [
                "items related with furniture decor interiors"
            ],
            "INSURANCE": [
                "related to insurance companyt and vehicle insurance"
            ],
            "IT & ACCESSORIES": [
                "Laptop, Keyboard, Mouse any accessory supply"
            ],
            "IT Expenses": [
                "Fortinet Fortigate 80F Unified Threat Protection",
                "Sales Order ERP Web Software development",
                "SAP Business One (bizhub)"
            ],
            "PLANT & MACHENERY": [
                "related to equipment used in a business to carry out operation"
            ],
            "PRINTING AND STATIONARY": [
                "related to books and stationary suppliers",
                "Crayons Corp Pvt. Ltd. (company Name)"
            ],
            "Rep Maint Exp-Pool A ": [
                "related to building, structure and similar works of permanent nature",
                "Auto or Repairing Workshop"
            ],
            "Repair and Maintainance Admin -Pool B": [
                "Electronics related repair and maintenance",
                "computers, data processing equipments, furiture, fixture and office equpments"
            ],
            "Telephones Expenses": [
                "SMS and call related invoice"
            ],
            "Travelling Expenses-Directors": [
                "related to hotel room expenses ",
                "Hotel names on vendor names",
                "(customer name: Atul Neupane)"
            ],
            "Travelling Expenses-Staffs": [
                "related to hotel room expenses ",
                "Hotel names on vendor names",
                "(customer name: Sabina, Mahesh)"
            ],
            "Travelling Expenses-Others": [
                "related to hotel room expenses",
                "Hotel names on vendor names",
                "(customer name: Sarbottam)"
            ],
            "Rep Maint Exp-Pool C": [
                "automobile, bus and minibus"
            ],
            "Repair and Maintainance Admin -Pool D": [
                "Construction and earth moving equipments, unabsorbed pollution control cost and any tangible assets not included in above blocks"
            ],
            "Rep Maint Exp-Pool E": [
                "Intangible assets (patents, copyrights, trade marks, software etc (cost+life down to which are not included in block D assets)"
            ],
            "SCRAP": [
            "Related to iron scraps or metal scraps",
            "Iron Scrap or Sponge Iron"
            ]
            }
        """
        invoice_data = json.loads(invoice_json_data)
        line_items = invoice_data["line_items"]
        vendor_name = invoice_data["vendor_details"]["name"]
        invoice_description = f"Invoice from vendor {vendor_name}"

        gl_mapping = json.loads(gl_mapping_json_data)

        # setup LangChain model
        model = self.client

        # define system + user prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that helps classify invoice line items for SAP G/L accounts.
            You will be provided with invoice details, a line item description, and a mapping of example descriptions to G/L accounts.
            Prioritize suggesting a G/L account from the provided mapping if the line item description is similar to any of the examples.
            If no similar example is found in the mapping, analyze the 'products' description, vendor name, and overall invoice description to suggest a relevant G/L account based on common accounting practices.
            Refer to the provided mapping of example descriptions to G/L accounts when making your suggestions.
            """),

            ("user", """Invoice Details:
            Vendor Name: {vendor_name}
            Invoice Description: {invoice_description}
            Line Item Products: {products}

            G/L Account Mapping:
            {gl_mapping_text}

            Classify this line item and suggest a G/L account from the mapping if applicable, otherwise suggest the most relevant G/L account.
            """)
        ])

        # for Langchain chain
        chain = prompt | model

        # to find G/L account from mapping
        def get_gl_from_mapping(products, mapping):
            for gl_account, descriptions in mapping.items():
                for desc in descriptions:
                    if desc.lower() in products.lower():
                        return gl_account
            return None

        classified_items = []
        for item in line_items:
            products = item.get("products")

            if products:
                # direct mapping first
                suggested_gl_account_raw = get_gl_from_mapping(products, gl_mapping)

                if suggested_gl_account_raw:
                    #  a match is found in the mapping, use it
                    item["suggested_gl_account"] = suggested_gl_account_raw
                    item["classification_source"] = "mapping"
                    classified_items.append(item)
                else:
                    # no match in mapping, Langchain for suggestion
                    try:
                        response = chain.invoke({
                            "products": products,
                            "vendor_name": vendor_name,
                            "invoice_description": invoice_description,
                            "gl_mapping_text": json.dumps(gl_mapping, indent=2)
                        })
                        suggested_gl_account_model = response.content.strip()
                        # Further process the string to extract the actual GL account name from the model's response
                        extracted_name = suggested_gl_account_model
                        if "**" in extracted_name:
                            extracted_name = extracted_name.split("**")[-2].strip() # Get the text between the last two **
                        else:
                            # Fallback if the format changes
                            extracted_name = extracted_name.split(".")[-1].strip()

                        item["suggested_gl_account"] = extracted_name
                        classified_items.append(item)

                    except Exception as e:
                        # to catch any exception during invoke and print details
                        logger.error(f"Error during Langchain invocation for products: {products}")
                        logger.error(f"Error type: {type(e).__name__}")
                        logger.error(f"Error message: {e}")
                        item["suggested_gl_account"] = f"Classification failed: {type(e).__name__}"
                        item["classification_source"] = "error"
                        classified_items.append(item)
            else:
                item["suggested_gl_account"] = "OTHER"
                item["classification_source"] = "no products"
                classified_items.append(item)
        return classified_items[0]['suggested_gl_account']


# # Get API key from environment variables
# api_key = settings.GOOGLE_API_KEY
# if not api_key:
#     logger.error("GOOGLE_API_KEY not found in environment variables")
#     raise ValueError("GOOGLE_API_KEY not found in environment variables")

# # Function to parse the raw JSON string
# def parse_invoice_json(raw_json_string: str) -> Optional[Dict]:
#     """Parses a raw string to extract and load a JSON object."""

#     json_start = raw_json_string.find('{')
#     json_end = raw_json_string.rfind('}')

#     if json_start != -1 and json_end != -1:
#         json_substring = raw_json_string[json_start : json_end + 1]
#         try:
#             invoice_data_dict = json.loads(json_substring)
#             return invoice_data_dict
#         except json.JSONDecodeError as e:
#             logger.error(f"Error decoding JSON: {e}")
#             return None
#     else:
#         logger.error("Could not find a valid JSON object in the string.")
#         return None

# # Function to get the Gemini model
# def get_gemini_model():
#     """Initializes and returns the ChatGoogleGenerativeAI model."""
#     logger.info("Initializing Gemini model...")
#     return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# # Function to classify the invoice
# def classify_invoice(invoice_json: dict, model) -> str:
#     """
#     Classifies an invoice based on its content using a Gemini model.
#     Also checks for a specific net_amount condition.
#     """
#     try:
#         total_amount = invoice_json.get("payment_details", {}).get("grand_total")

#         if total_amount is not None:
#             try:
#                 cleaned_amount_str = re.sub(r'[^0-9.]', '', str(total_amount))
#                 total_amount_float = float(cleaned_amount_str)

#                 if total_amount_float < 2000:
#                     logger.info("Identified as outgoing_payment based on grand_total < 2000")
#                     return "outgoing_payment"
#             except (ValueError, TypeError) as num_err:
#                     logger.error(f"Could not convert grand_total to number: {total_amount} - {num_err}")
#                     # Continue to model classification if number conversion fails
#     except Exception as e:
#         # Catch any unexpected errors during the initial check
#         logger.error(f"Error during grand_total check: {e}")
#     # If the net_amount condition is not met, use the model for classification
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """You are a document classifier for SAP systems. Based on the invoice JSON data, classify the document into one of the following types:
# - ap_invoice
# - ap_invoice_with_lc

# Your classification should depend on the nature of the document:
# - 'ap_invoice' is a standard vendor invoice
# - 'ap_invoice_with_lc' includes reference to LC (Letter of Credit) in payment mode, particulars, or vendor behavior (if invoice_details.lc_no is preset it is ap_invoice_with_lc)

# Respond with only one of the labels: ap_invoice or ap_invoice_with_lc.
# """),
#         ("user", "Here is the invoice data:\n{invoice_json}")
#     ])

#     chain = prompt | model

#     try:
#         response = chain.invoke({
#             "invoice_json": json.dumps(invoice_json, indent=2)
#         })
#         return response.content.strip()
#     except Exception as e:
#         logger.error(f"Classification failed: {type(e).__name__} - {e}")
#         return f"Classification failed: {type(e).__name__} - {e}"

# classifier_agent = Classifier()

# # Define the raw JSON string
# raw_json = collection.find_one({"uid": total_uploads},{"extracted_details":1, "_id": 0 })
# raw_json_string = json.dumps(raw_json) 

# # Parse the JSON and classify the invoice
# invoice_data_dict = classifier_agent.parse_invoice_json(raw_json_string)

# if invoice_data_dict:
#     model = classifier_agent.client
#     classification_result = classifier_agent.classify_invoice(invoice_data_dict, model)

# def match_vendor_name(document_id: int):
#     try:
#         API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/sentence-similarity"
#         access_token = os.getenv("HF_API_KEY")
#         headers = {
#             "Authorization": f"Bearer {access_token}",
#         }
#         df = pd.read_csv("backend/assets/sap_fields.csv")
#         logger.info("Loaded vendor names from CSV.")
#         sap_fields = df.values.tolist()

#         def query(payload):
#             response = requests.post(API_URL, headers=headers, json=payload)
#             return response.json()
        
#         document = collection.find_one({"uid": document_id}, {"extracted_details": 1, "_id": 0})
#         vendor_name = document.get("extracted_details", {}).get("vendor_details", {}).get("name")
#         vendor_name_list = [list[0] for list in sap_fields]

#         output = query({
#             "inputs": {
#                 "source_sentence": vendor_name,
#                 "sentences": vendor_name_list
#             }
#         })

#         if output and isinstance(output, list):
#             scores = output[0] if isinstance(output[0], list) else output
            
#             if not isinstance(scores, list):
#                 logger.error(f"Unexpected scores format: {type(scores)}")
#                 return 

#             highest_score = max(scores)
#             best_index = scores.index(highest_score)
            
#             best_match = vendor_name_list[best_index]
                        
#             logger.info("Best match found while classifying")
#             logger.info(f"Vendor name: {best_match}")
#             logger.info(f"Similarity score: {highest_score:.2%}")
#             logger.info(f"Index in list: {best_index}")

#             collection.update_one(
#                 {"uid": document_id},
#                 {"$set": {"extracted_details.vendor_details.name": best_match}}
#             )
#             logger.info(f"Vendor name updated in database for document ID {document_id} to {best_match}.")
#         else:
#             logger.error(f"Invalid API response format: {type(output)}")
#             logger.error(f"Response content: {output}")

#     except Exception as e:
#         logger.error(f"An unexpected error occurred during vendor name matching for document ID {document_id}: {e}")
#         return f"An internal error occurred: {str(e)}"


# def process_classification(document_id: int):
#     """Fetches document by ID and processes classification based on extracted details."""
#     logger.info("Attempting to process classification for document ID: {document_id}")
#     try:
#         # Fetch the document from MongoDB using the uid
#         document = collection.find_one({"uid": document_id}, {"extracted_details": 1, "_id": 0})

#         if not document:
#             logger.error(f"Document with ID {document_id} not found in database.")
#             return "Document not found."

#         # Directly access the extracted_details field from the document dictionary
#         extracted_details = document.get("extracted_details")

#         if not extracted_details:
#             logger.error(f"No 'extracted_details' field found for document ID: {document_id}")
#             return "No extracted details found for this document."

#         # Check if extracted_details is a string (as it sometimes might be stored)
#         # If it is a string, try to parse it as JSON
#         if isinstance(extracted_details, str):
#             try:
#                 invoice_data_dict = json.loads(extracted_details)
#             except json.JSONDecodeError as e:
#                 logger.error(f"Error decoding JSON from 'extracted_details' for document ID {document_id}: {e}")
#                 return f"Error processing extracted details: Invalid JSON format ({e})"
#         else:
#             # Assume it's already the correct dictionary format
#             invoice_data_dict = extracted_details
#             logger.info(f"'extracted_details' for document ID {document_id} is already a dictionary.")

#         # Ensure we have a dictionary before passing to classify_invoice
#         if not isinstance(invoice_data_dict, dict):
#              logger.info(f"'extracted_details' for document ID {document_id} is not in a valid dictionary format after processing.")
#              return "Extracted details are not in a valid dictionary format for classification."

#         # Get the Gemini model
#         model = get_gemini_model()

#         # Classify the invoice
#         classification_result = classify_invoice(invoice_data_dict, model)

#         if classification_result == 'ap_invoice':
#             gl_classified = gl_account_classifier(document_id)
#             collection.update_one({"uid": document_id}, {"$set":  {"gl_classification": gl_classified}})

#         else: 
#             logger.info("Not a ap_invoice")

#         return classification_result

# except Exception as e:
#     logger.error(f"An unexpected error occurred during classification for document ID {document_id}: {e}")
#     return f"An internal error occurred: {str(e)}"

# def gl_account_classifier(document_id:id):
#     document = collection.find_one({"uid": document_id}, {"extracted_details": 1, "_id": 0})
#     raw_vendor_extracted_details = document.get("extracted_details", {})
#     vendor_extracted_details = json.dumps(raw_vendor_extracted_details)
    
#     invoice_json_data = vendor_extracted_details
#     gl_mapping_json_data = """
#         {
#         "Advertisement Expenses": [
#             "Sales-KTV ( 5 sec Headline break all news Sarbottam steelTVC cost of KrV dated Magh 1-30 2081 ) (As per RO)",
#             "Toward the Cost of Facebook Page Management",
#             "Advertisement tax and service",
#             "Radio Advertising",
#             "Volume Branding"
#         ],
#         "CARGO FEE": [
#             "Consignment Note",
#             "DHL Express or any cargo company",
#             "Cargo and Courier"
#         ],
#         "Cleaning Expenses": [
#             "Harpic Dettol Lizol Exo Odonil (bhatbhateni)"
#         ],
#         "Electricity Expenses": [
#             "related to energy companies (electricit charges of a certain month in line item)"
#         ],
#         "FURNITURE & FIXTURE": [
#             "items related with furniture decor interiors"
#         ],
#         "INSURANCE": [
#             "related to insurance companyt and vehicle insurance"
#         ],
#         "IT & ACCESSORIES": [
#             "Laptop, Keyboard, Mouse any accessory supply"
#         ],
#         "IT Expenses": [
#             "Fortinet Fortigate 80F Unified Threat Protection",
#             "Sales Order ERP Web Software development",
#             "SAP Business One (bizhub)"
#         ],
#         "PLANT & MACHENERY": [
#             "related to equipment used in a business to carry out operation"
#         ],
#         "PRINTING AND STATIONARY": [
#             "related to books and stationary suppliers",
#             "Crayons Corp Pvt. Ltd. (company Name)"
#         ],
#         "Rep Maint Exp-Pool A ": [
#             "related to building, structure and similar works of permanent nature",
#             "Auto or Repairing Workshop"
#         ],
#         "Repair and Maintainance Admin -Pool B": [
#             "Electronics related repair and maintenance",
#             "computers, data processing equipments, furiture, fixture and office equpments"
#         ],
#         "Telephones Expenses": [
#             "SMS and call related invoice"
#         ],
#         "Travelling Expenses-Directors": [
#             "related to hotel room expenses ",
#             "Hotel names on vendor names",
#             "(customer name: Atul Neupane)"
#         ],
#         "Travelling Expenses-Staffs": [
#             "related to hotel room expenses ",
#             "Hotel names on vendor names",
#             "(customer name: Sabina, Mahesh)"
#         ],
#         "Travelling Expenses-Others": [
#             "related to hotel room expenses",
#             "Hotel names on vendor names",
#             "(customer name: Sarbottam)"
#         ],
#         "Rep Maint Exp-Pool C": [
#             "automobile, bus and minibus"
#         ],
#         "Repair and Maintainance Admin -Pool D": [
#             "Construction and earth moving equipments, unabsorbed pollution control cost and any tangible assets not included in above blocks"
#         ],
#         "Rep Maint Exp-Pool E": [
#             "Intangible assets (patents, copyrights, trade marks, software etc (cost+life down to which are not included in block D assets)"
#         ],
#         "SCRAP": [
#            "Related to iron scraps or metal scraps",
#            "Iron Scrap or Sponge Iron"
#         ]
#         }
#     """
#     invoice_data = json.loads(invoice_json_data)
#     line_items = invoice_data["line_items"]
#     vendor_name = invoice_data["vendor_details"]["name"]
#     invoice_description = f"Invoice from vendor {vendor_name}"

#     gl_mapping = json.loads(gl_mapping_json_data)

#     # setup LangChain model
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

#     # define system + user prompt
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """You are an AI assistant that helps classify invoice line items for SAP G/L accounts.
#         You will be provided with invoice details, a line item description, and a mapping of example descriptions to G/L accounts.
#         Prioritize suggesting a G/L account from the provided mapping if the line item description is similar to any of the examples.
#         If no similar example is found in the mapping, analyze the 'products' description, vendor name, and overall invoice description to suggest a relevant G/L account based on common accounting practices.
#         Refer to the provided mapping of example descriptions to G/L accounts when making your suggestions.
#         """),

#         ("user", """Invoice Details:
#         Vendor Name: {vendor_name}
#         Invoice Description: {invoice_description}
#         Line Item Products: {products}

#         G/L Account Mapping:
#         {gl_mapping_text}

#         Classify this line item and suggest a G/L account from the mapping if applicable, otherwise suggest the most relevant G/L account.
#         """)
#     ])

#     # for Langchain chain
#     chain = prompt | model

#     # to find G/L account from mapping
#     def get_gl_from_mapping(products, mapping):
#         for gl_account, descriptions in mapping.items():
#             for desc in descriptions:
#                 if desc.lower() in products.lower():
#                     return gl_account
#         return None

#     classified_items = []
#     for item in line_items:
#         products = item.get("products")

#         if products:
#             # direct mapping first
#             suggested_gl_account_raw = get_gl_from_mapping(products, gl_mapping)

#             if suggested_gl_account_raw:
#                 #  a match is found in the mapping, use it
#                 item["suggested_gl_account"] = suggested_gl_account_raw
#                 item["classification_source"] = "mapping"
#                 classified_items.append(item)
#             else:
#                 # no match in mapping, Langchain for suggestion
#                 try:
#                     response = chain.invoke({
#                         "products": products,
#                         "vendor_name": vendor_name,
#                         "invoice_description": invoice_description,
#                         "gl_mapping_text": json.dumps(gl_mapping, indent=2)
#                     })
#                     suggested_gl_account_model = response.content.strip()
#                     # Further process the string to extract the actual GL account name from the model's response
#                     extracted_name = suggested_gl_account_model
#                     if "**" in extracted_name:
#                         extracted_name = extracted_name.split("**")[-2].strip() # Get the text between the last two **
#                     else:
#                         # Fallback if the format changes
#                         extracted_name = extracted_name.split(".")[-1].strip()

#                     item["suggested_gl_account"] = extracted_name
#                     classified_items.append(item)

#                 except Exception as e:
#                     # to catch any exception during invoke and print details
#                     logger.error(f"Error during Langchain invocation for products: {products}")
#                     logger.error(f"Error type: {type(e).__name__}")
#                     logger.error(f"Error message: {e}")
#                     item["suggested_gl_account"] = f"Classification failed: {type(e).__name__}"
#                     item["classification_source"] = "error"
#                     classified_items.append(item)
#         else:
#             item["suggested_gl_account"] = "OTHER"
#             item["classification_source"] = "no products"
#             classified_items.append(item)
#     return classified_items[0]['suggested_gl_account']
