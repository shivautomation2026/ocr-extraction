from fastapi import APIRouter, HTTPException
from backend.services.sap_api import SAPClient
from pydantic import BaseModel
import logging

router = APIRouter(prefix="/sap", tags=["SAP API"])
sap_client = SAPClient()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLine(BaseModel):
    ItemCode: str
    UoMEntry: int
    TaxCode: str
    Quantity: int
    UnitPrice: float

class SAPPurchaseInvoice(BaseModel):
    CardCode: str
    DocDate: str
    DocumentLines: list[DocumentLine]

@router.post("/PurchaseInvoices", summary="Post Purchase Invoice to SAP", description="Post a purchase invoice to SAP Business One system.")
async def post_purchase_invoice(invoice: SAPPurchaseInvoice):
    try:
        logger.info(f"Received data: {invoice}")
        payload = invoice.model_dump(exclude_none=True)

        result = sap_client.post_purchase_invoice(payload)
        if result:
            if result.get("error"):
                raise HTTPException(
                    status_code=result.get("status_code", 400), 
                    detail=f"SAP Error: {result.get('detail', 'Unknown error')}"
                )
            logger.info("DocEntry: %s", result.get("DocEntry"))
            return {"status": "success", "DocEntry": result.get("DocEntry"), "data": result}
        else:
            raise HTTPException(status_code=400, detail="Failed to post purchase invoice to SAP.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error posting purchase invoice")
        raise HTTPException(status_code=500, detail=str(e))