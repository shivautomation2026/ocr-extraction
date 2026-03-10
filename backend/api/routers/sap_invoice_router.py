from fastapi import APIRouter, Depends, HTTPException
from backend.database import init_db
from backend.services.sap_api import sap_client
from pydantic import BaseModel
import logging
from backend.core.config import settings
import requests

router = APIRouter(prefix="/sap", tags=["SAP API"])

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

class SAPLoginRequest(BaseModel):
    CompanyDB: str
    UserName: str
    Password: str

@router.post("/Login",summary="Login to SAP", description="Authenticate with SAP Business One system.")
async def sap_login(creds = Depends(SAPLoginRequest)):
    base_url = settings.BASE_URL
    session = requests.Session()

    try:
        login_req = session.post(f"{base_url}Login", json=creds, verify=False)

        if login_req.status_code == 200:
            logger.info("Successfully logged in to SAP.")
        else:
            logger.error(f"Login failed with status code: {login_req.status_code}")
            exit(1)
    except requests.exceptions.RequestException as e:
            print(f"Login request failed: {e}")
            exit(1)
    

@router.post("/PurchaseInvoices", summary="Post Purchase Invoice to SAP", description="Post a purchase invoice to SAP Business One system.")
async def post_purchase_invoice(document_id: int, collection=Depends(init_db)):
    try:
        mapped_fields = await collection.find_one({"uid": document_id}, {"mapped_result": 1, "_id": 0})
        
        if not mapped_fields or "mapped_result" not in mapped_fields:
            raise HTTPException(status_code=404, detail="Mapped fields not found for the given document ID.")
        
        raw_lines = mapped_fields["mapped_result"].get("DocumentLines", [])
        keep = ['ItemCode', 'UoMEntry', 'TaxCode', 'Quantity', 'UnitPrice']
        document_lines = [
            {k: int(v) if k == 'UoMEntry' else v for k, v in line.items() if k in keep}
            for line in raw_lines
        ]
        invoice = {
            "CardCode": mapped_fields["mapped_result"].get("CardCode"),
            "DocDate": mapped_fields["mapped_result"].get("DocDate"),
            "DocumentLines": document_lines
        }
        
        result = sap_client.post_purchase_invoice(invoice)
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