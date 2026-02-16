from fastapi import Request, UploadFile, Form, APIRouter
from fastapi.responses import JSONResponse, FileResponse
import os
import shutil
from backend.services.ocr_processor import OCR_Processor
from datetime import datetime
from backend.database import collection
import logging
from bson import ObjectId
import pandas as pd
import tempfile
from backend.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/extract", tags=["Text Extraction"])

current_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(current_dir, "temp_files")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def format_datetime(dt):
    """Format datetime to a readable string"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


@router.post("/", summary = "Upload and extract text from files", description="Upload up to 5 PDF files and optionally provide a custom prompt for text extraction. The extracted text and structured content will be returned in the response.")
async def upload_file(request: Request, file_list: list[UploadFile], prompt: str = Form(None)):
    try:
        # Create a new OCR client for each request to get fresh cost tracker
        ocr_client = OCR_Processor()
        save_as_excel = False
        # Limit uploads to maximum 5 files
        if len(file_list) > 5:
            logger.error(f"Upload attempt with {len(file_list)} files, exceeding the limit (5).")
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Maximum 5 files allowed per upload. You attempted to upload {len(file_list)} files."
                }
            )
        
        uploads = []
        document_ids = []
        for file in file_list:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            result = ocr_client.process_file(file_path, prompt or "")
            total_uploads = collection.count_documents({"file_name": {"$exists":True}})

            llm_cost_data = ocr_client.cost_tracker.get_total_usage()
            
            if prompt:
                structure = {
                    "file_name": file.filename,
                    "uid": total_uploads+1,
                    "prompt_type": "user_given_prompt",
                    "prompt": prompt,
                    "raw_text": result.extracted_text,
                    "extracted_details": result.content,
                    "llm_cost_tracking": {
                        "model": llm_cost_data["model"],
                        "total_input_tokens": llm_cost_data["total_input_tokens"],
                        "total_output_tokens": llm_cost_data["total_output_tokens"],
                        "total_cost": llm_cost_data["total_cost"],
                        "usage_records": ocr_client.cost_tracker.usage_records,
                        "tracked_at": format_datetime(datetime.now())
                    },
                    "uploaded_at": format_datetime(datetime.now())
                }
            else:
                structure = {
                    "file_name": file.filename,
                    "uid": total_uploads+1,
                    "prompt_type": "default_prompt",
                    "raw_text": result.extracted_text,
                    "extracted_details": result.content,
                    "llm_cost_tracking": {
                        "model": llm_cost_data["model"],
                        "total_input_tokens": llm_cost_data["total_input_tokens"],
                        "total_output_tokens": llm_cost_data["total_output_tokens"],
                        "total_cost": llm_cost_data["total_cost"],
                        "usage_records": ocr_client.cost_tracker.usage_records,
                        "tracked_at": format_datetime(datetime.now())
                    },
                    "uploaded_at": format_datetime(datetime.now())
                }
            inserted_doc = collection.insert_one(structure)
            document_ids.append(inserted_doc.inserted_id)
            ocr_client.cost_tracker.reset()
            
            os.remove(file_path)

            if result.status == "success":
                document = collection.find_one({"_id": ObjectId(inserted_doc.inserted_id)})
                document_id = document["uid"]
                save_as_excel = True

                uploads.append({
                    "file_name": file.filename,
                    "document_id": document_id,
                    "content": result.content,
                    "extracted_text": result.extracted_text,
                })
                
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": result.message
                    }
                )
            
            # if save_as_excel:
            #     try:
            #         excel_data = []
            #         doc = collection.find_one({"_id": ObjectId(inserted_doc.inserted_id)})
            #         if doc:

                

        return JSONResponse(
                    status_code=200,
                    content={
                        "status": "success",
                        "message": "Text extracted and structured successfully",
                        "data": uploads
                    }
                )   

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )
        # Prepare response

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@router.get("/text-extraction/pdf")
async def get_all_extractions(file:str = None):
    try:
        if file:
            if not file.endswith('.pdf'):
                filename = f"{file}.pdf"
                document_find = list(collection.find(({"file_name": filename}), {'_id': 0}))
                return {'documents': document_find}
            else:
                document_find = list(collection.find(({"file_name": file}), {'_id': 0}))
                return {'documents': document_find}
        else:
            all_extractions = list(collection.find(({"file_name":{"$exists": True}}),{'_id':0,'default_prompt':0}))
            # for item in all_extractions:
            #     item['_id'] = str(item['_id'])
            return {'documents': all_extractions}
    except Exception as e:
        return{
            "message": f"Error retrieving documents: {str(e)}"
        }

@router.delete("/delete/{file}")
async def delete_extraction(file: str):
    try:
        if not file.endswith('.pdf'):
            filename = f"{file}.pdf"
        result = collection.delete_one({"file_name": filename})
        if result.deleted_count == 0:
            return JSONResponse(
                status_code=404,
                content={"message": f"No file found with name: {file}"}
            )
        return JSONResponse(
            status_code=200,
            content={"message": f"File {file}.pdf deleted successfully"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error deleting file: {str(e)}"}
        )

# @router.get("/sap_fields")
# async def get_sap_fields():
#     # base_url = settings.SAP["BASE_URL"]
#     try:
#         base_url = "https://202.79.47.181:50000/b1s/v1/"
#         if login(base_url):
#             print("Login successful.")
#             save_items_to_csv(base_url)
#             save_item_groups_to_csv(base_url)
#             save_uom_groups_to_csv(base_url)
#             print("Data extraction completed.")
#             return JSONResponse(
#                 status_code=200,
#                 content={"message": "SAP fields extracted and saved to CSV files successfully."}
#             )
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"message": f"Error extracting SAP fields: {str(e)}"}
#         )