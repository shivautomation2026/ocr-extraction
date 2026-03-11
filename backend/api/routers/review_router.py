from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import JSONResponse
from backend.database import init_db
from pydantic import BaseModel, Field
from typing import Annotated, Any, Dict, List
import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/review", tags=["Review"])

class VendorDetails(BaseModel):
    name: str | None = None
    address: str | None = None
    contact_number: str | None = None
    email: str | None = None
    website: str | None = None
    pan_number: int | None = None

class CustomerDetails(BaseModel):
    name: str | None = None
    address: str | None = None
    contact_number: str | None = None
    pan_number: int | None = None

class InvoiceDetails(BaseModel):
    bill_number: str | None = None
    posting_date: str | None = None
    bill_date: str | None = None
    nepali_miti: str | None = None
    mode_of_payment: str | None = None
    finance_manager: str | None = None
    authorized_signatory: str | None = None
    lc_no: str | None = None

class PaymentDetails(BaseModel):
    net_amount: float | None = None
    discount_amount: float | None = None
    taxable_amount: float | None = None
    vat_percentage: str | None = None
    vat_amount: float | None = None
    grand_total: float | None = None
    grand_total_in_words: str | None = None

class LineItem(BaseModel):
    hs_code: str | None = None
    description: str | None = None
    quantity: float | None = None
    rate: float | None = None
    amount: float | None = None

class LineItemUpdate(BaseModel):
    index: int | None = None
    data: LineItem  
        
class UpdateExtractedDetails(BaseModel):
    vendor_details: VendorDetails | None = None
    customer_details: CustomerDetails | None = None
    invoice_details: InvoiceDetails | None = None
    payment_details: PaymentDetails | None = None
    line_items: List[LineItemUpdate] | None = None


@router.get("/{document_id}", summary="Get document details for review", description="Fetches all relevant document data for a given document ID to facilitate the review and approval process.")
async def get_review(document_id: int, collection=Depends(init_db)):
    """
    Returns all relevant document data so a reviewer can inspect and decide on approval.
    """
    try:
        document = await collection.find_one(
            {"uid": document_id},   
            {"_id": 0, "extracted_details": 1, "mapped_result": 1, "approval": 1, "file_name": 1, "uploaded_at": 1}
        )

        if not document:
            raise HTTPException(status_code=404, detail=f"Document with UID {document_id} not found.")

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "document_uid": document_id,
                "file_name": document.get("file_name"),
                "uploaded_at": document.get("uploaded_at"),
                "approval": document.get("approval", "pending"),
                "extracted_details": document.get("extracted_details"),
                "mapped_result": document.get("mapped_result"),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving review for document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}/approve", summary="Approve document", description="Set approval status to 'approved' after reviewing the document data.")
async def approve_document(document_id: int, collection=Depends(init_db)):
    """
    Marks the document as approved by setting its approval status to 'approved'.
    """
    try:
        document = await collection.find_one({"uid": document_id}, {"_id": 0, "approval": 1})
        
        if document and document.get("approval") == "approved":
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": "Document is already approved.",
                    "document_uid": document_id,
                    "approval": "approved",
                }
            )
            
        result = await collection.update_one(
            {"uid": document_id},
            {"$set": {"approval": "approved"}}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail=f"Document with UID {document_id} not found.")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Document approved successfully.",
                "document_uid": document_id,
                "approval": "approved",
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/{document_id}/edit", summary="Confirm document approval", description="Set approval status to 'confirmed' after reviewing the document data.")
async def edit_details(document_id: int, update_fields: Annotated[UpdateExtractedDetails, Body()], collection=Depends(init_db)):
    """
    Marks the document as approved by setting its approval status to 'confirmed'.
    """
    try:
        document = await collection.find_one({"uid": document_id}, {"_id": 0})
        
        if document and document.get("approval") == "confirmed":
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": "Document is already confirmed.",
                    "document_uid": document_id,
                    "approval": "confirmed",
                }
            )
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document with UID {document_id} not found.")
        
        fields = {}
        dumped = update_fields.model_dump(exclude_none=True)

        for section in ("vendor_details", "customer_details", "invoice_details","payment_details"):
            if section in dumped:
                for field, val in dumped[section].items():
                    fields[f"extracted_details.{section}.{field}"] = val
        
        if update_fields.line_items:
            doc = await collection.find_one(
            {"uid": document_id},
            {"_id": 0, "extracted_details.line_items": 1}
            )
            
            if not doc:
                raise HTTPException(status_code=404, detail="Invoice not found")
            
            if not doc.get("extracted_details", {}).get("line_items"):
                raise HTTPException(status_code=400, detail="Document has no line items to update")

            current_items = doc.get("extracted_details", {}).get("line_items", [])
            
            for update in update_fields.line_items:
                idx = update.index

                # Validate index
                if idx < 0 or idx >= len(current_items):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Line item index {idx} is out of range. "
                            f"Document has {len(current_items)} line items (0 to {len(current_items)-1})."
                    )

                # Merge only provided fields into the existing line item
                changes = update.data.model_dump(exclude_none=True)
                for field, val in changes.items():
                    fields[f"extracted_details.line_items.{idx}.{field}"] = val
                    
        if not fields:
            raise HTTPException(status_code=400, detail="No fields provided to update")

        result = await collection.update_one(
            {"uid": document_id},
            {"$set": fields}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Invoice not found")
        
        updated = await collection.find_one(
            {"uid": document_id},
            {"_id": 0, "extracted_details": 1}
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Waiting for final approval. Document details updated successfully.",
                "document_uid": document_id,
                "updated_extracted_details": updated.get("extracted_details", {}),
                "approval": document.get("approval")
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error confirming approval for document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


