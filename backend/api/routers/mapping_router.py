from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import datetime
# from backend.database import collection
import logging
import pandas as pd
import json
from google import genai
from backend.core.config import settings
from backend.services.mapping import Mapper
from backend.utils.cost_tracker import LLMCostTracker, extract_langchain_usage
from pydantic import BaseModel, Field
from typing import Optional
from backend.database import init_db

router = APIRouter(prefix="/mapping", tags=["Field Mapping"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLine(BaseModel):
    ItemCode: str
    TaxCode: str
    UoMEntry: int

class SAPFields(BaseModel):
    CardCode: str
    DocDate: str
    DocumentLines: list[DocumentLine]

class DocumentLinePatch(BaseModel):
    line_index: int
    ItemCode: Optional[str] = None
    TaxCode: Optional[str] = None
    UoMEntry: Optional[int] = None

class SAPFieldsPatch(BaseModel):
    CardCode: Optional[str] = None
    DocDate: Optional[str] = None
    DocumentLines: Optional[list[DocumentLinePatch]] = None

try:
    sap_required_fields_df = pd.read_csv('backend/assets/sap invoice required field details.csv')
    cleaned_df = sap_required_fields_df.dropna(how='all')
    logger.info("Successfully loaded CSV mapping table.")
except FileNotFoundError:
    logger.error("FATAL: 'sap invoice required field details.csv' not found.")
    cleaned_df = None

try:
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in the environment.")
    # client = genai.Client(vertexai=True, project=settings.GOOGLE_CLOUD_PROJECT, location=settings.GOOGLE_CLOUD_LOCATION)
    client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    
    logger.info("Successfully configured Gemini client with model settings.")
except Exception as e:
    logger.error(f"FATAL: Failed to configure Gemini client: {e}")
    client = None

item_mapper = Mapper()

@router.get("/get-mappings/{document_uid}", summary="Get field mappings", description="Retrieve field mappings for a given document type.")
async def get_field_mappings(document_uid: int, collection = Depends(init_db)):
    approval_status = await collection.find_one({"uid": document_uid}, {"approval": 1, "_id": 0})
    if not approval_status or approval_status.get("approval") != "approved":
        raise HTTPException(status_code=403, detail="Document is not approved for mapping. Current status: {}".format(approval_status.get("approval") if approval_status else "unknown"))
    if cleaned_df is None:
        raise HTTPException(status_code=503, detail="Mapping service is unavailable: CSV file not loaded.")
    if client is None:
        raise HTTPException(status_code=503, detail="Mapping service is unavailable: AI client not configured.")
    # try:
    await item_mapper.find_similar_vendor(document_uid)

    await item_mapper.map_items_to_codes(document_uid)
    try:
        document_json = await collection.find_one({"uid": document_uid}, {"_id": 0, "extracted_details": 1})
        if not document_json:
            raise HTTPException(status_code=404, detail=f"Document with UID {document_uid} not found.")

        incoming_json = document_json.get("extracted_details", {})
        if not incoming_json:
            raise HTTPException(status_code=404, detail="No 'extracted_details' found in the document.")

        logger.info(f"Fetched document for UID {document_uid}.")

        prompt = f"""
        You are a data mapping assistant.

        Your task:
        Map fields from the following JSON invoice to SAP field names using the provided CSV mapping table.

        ### Incoming JSON
        {json.dumps(incoming_json, indent=2)}

        ### Mapping CSV
        Document name,Table name,sap field name
        {cleaned_df.to_string(index=False)}

        ### Rules
        1. Match each CSV "Document name" to the corresponding "sap field name" in the input JSON (case-insensitive, search recursively).
        2. "DocumentLines" should be an array constructed from all items in the line_items.
        3. For each item in line_items:
           - If "ItemCode" exists in the line item, use it directly
           - If "ItemCode" doesn't exist but "product" or "description" exists, map it using the ItemCode field
           - Include the Quantity in float format.
           - Include the UnitPrice in float format.
        4. Ignore any mention of table names — they are not needed.
        5. Use the "sap field name" from the CSV as the output key.
        6. If "vat_percentage" exists and is "13", set "TaxCode" to "VAT13".
           - Otherwise, set "TaxCode" to "VAT13".
        7. If "mode_of_payment" exists in incoming json then refer to it as "transaction type"
        8. DocDate is {datetime.date.today().strftime("%Y-%m-%d")}
        9. If any mapped field is missing in the JSON, include it with an empty string.
        10. Return the result as a single flat JSON object.
        11. Do NOT include markdown, code fences (```), comments, or any text outside of the final JSON object.

        ### Output format (STRICT JSON)
        Return output as plain JSON like this example:
        {{
        "CardName": "...",
        "CardCode": "...",
        "DocDate": "...",
        "DocumentLines": [
        {{
            "ItemCode": "EL00347",
            "Description": "AUXILLARY CONTACT BLOCK 3 RT 2015 HA",
            "Quantity": 100, // in float
            "TaxCode": "VAT13",
            "UnitPrice": 50, // in float
        }}
    ]
        }}
        Important: Return ONLY the standard JSON schema object without any markdown formatting or code blocks. No explanations, no headings, no extra text.

        """
        logger.info("Generating content with Gemini...")
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL_NAME,
            contents = prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )

        # token extraction for genai SDK  prompt_token_count/candidates_token_count
        usage = extract_langchain_usage(response)
        
        mapping_cost_tracker = LLMCostTracker(model_name=settings.GEMINI_MODEL_NAME)
        mapping_cost_tracker.track_llm_usage(
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            operation="field_mapping"
        )
        
        # Update existing cost tracking in MongoDB (append to existing)
        now = datetime.datetime.now()
        day = now.strftime("%Y-%m-%d")
        month = now.strftime("%Y-%m")
        
        existing_doc = await collection.find_one({"uid": document_uid}, {"llm_cost_tracking": 1})
        if existing_doc and "llm_cost_tracking" in existing_doc:

            await collection.update_one(
                {"uid": document_uid},
                {
                    "$inc":{
                        "llm_cost_tracking.total_input_tokens": usage["input_tokens"],
                        "llm_cost_tracking.total_output_tokens": usage["output_tokens"],
                        "llm_cost_tracking.total_cost": mapping_cost_tracker.total_cost
                    },
                    "$push":{
                        "llm_cost_tracking.usage_records": {"$each": mapping_cost_tracker.usage_records}
                    },
                    "$set":{
                        "llm_cost_tracking.last_updated": now.isoformat()
                }}
            )
            logger.info(f"Updated LLM cost tracking for document {document_uid}. Mapping cost: ${mapping_cost_tracker.total_cost:.6f}")
        else:
            mapping_cost_tracker.save_to_mongodb(document_uid)

        db = collection.database
        aggregation_collection = db["llm_cost_aggregation"]

        aggregations = [
            ("daily", f"daily_{day}", day),
            ("monthly", f"monthly_{month}", month),
            ("overall","overall", None)
        ]

        for agg_name, agg_id, agg_date in aggregations:
            update_doc = {
                "$inc":{
                    "total_input_tokens": usage["input_tokens"],
                    "total_output_tokens": usage["output_tokens"],
                    "total_cost": mapping_cost_tracker.total_cost,
                    "document_count": 1
                },
                "$set":{
                    "agg_type": agg_name,
                    "updated_at": now
                },
                "$setOnInsert":{
                    "created_at": now
                }
            }

            if agg_date:
                update_doc["$set"]["agg_date"] = agg_date


            await aggregation_collection.update_one(
                {"uid": agg_id},
                update_doc,
                upsert=True
                )
            
        logger.info(f"Updated aggregation collection with field mapping costs")

        if not response.text or not response.text.strip():
            logger.error("Gemini API returned an empty response.")
            raise HTTPException(status_code=500, detail="AI model returned an empty response.")

        logger.info("Successfully received response from Gemini.")
        mapped_result = json.loads(response.text)
        await collection.update_one(
            {"uid": document_uid},
            {"$set": {"mapped_result": mapped_result}}
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "document_uid": document_uid,
                "original_extracted_data": incoming_json,
                "mapped_result": mapped_result,
                "message": "Document mapped successfully. Please review before posting to SAP."
            }
        )

    except json.JSONDecodeError as e:
        logger.error(f"JSON Parsing Error: {e}. Raw AI Response: '{response.text}'")
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response. Raw text: {response.text}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during mapping: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
