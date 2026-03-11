from fastapi import APIRouter, HTTPException, Depends
from backend.utils.cost_queries import CostQueries
from backend.models import CostStatsRequest

router = APIRouter(prefix="/cost-tracking", tags=["LLM Cost Tracker"])

@router.get("/stats")
async def get_cost_stats(params: CostStatsRequest = Depends()):

    try:
        cost_queries = CostQueries()

        result = await cost_queries.get_stats(
            scope=params.scope,
            date=params.date,
            month=params.month,
            start_date=params.start_date,
            end_date=params.end_date,
            days=params.days,
            document_id=params.document_id
        )

        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")