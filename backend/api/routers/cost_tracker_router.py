from fastapi import APIRouter, HTTPException
from backend.models import CostStatsRequest


router = APIRouter(prefix="/cost-tracking", tags=["LLM Cost Tracker"])

@router.get("/stats")
def get_cost_stats(
    request: CostStatsRequest):

    try:
        from backend.database import collection

        result = get_cost_stats(
            collection=collection,
            scope=request.scope,
            date=request.date,
            month=request.month,
            start_date=request.start_date,
            end_date=request.end_date,
            days=request.days,
            document_id=request.document_id
        )

        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")