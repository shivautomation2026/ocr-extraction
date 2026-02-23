from datetime import datetime, timedelta
from typing import Optional
from backend.database import db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CostQueries:

    VALID_SCOPES = ["overall", "daily", "monthly", "range", "document"]

    def __init__(self):
        self.llm_cost_collection = db["documents"]
        self.aggregation_collection = db["llm_cost_aggregation"]
        self.now = datetime.now()


    def _empty_response(self, scope: str, **kwargs) -> dict:
        """ helper function to return empty response"""
        base = {
            "scope": scope,
            "total_cost": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "document_count": 0,
            "created_at": None,
            "updated_at": None
        }

        base.update(kwargs)
        return base
    
    
    def _format_aggregation(self, scope: str, record: dict, **kwargs) -> dict:
        """helper function for formatting aggregation result"""
        return{
            "scope": scope,
            "total_cost": record.get('total_cost', 0),
            "total_input_tokens": record.get('total_input_tokens', 0),
            "total_output_tokens": record.get('total_output_tokens', 0),
            "document_count": record.get('document_count', 0),
            "created_at": record.get('created_at', 0),
            "updated_at": record.get('updated_at', 0),
            **kwargs
        }


    def get_overall_stats(self)-> dict:
        """Get all time stats of document processed"""
        try:
            overall_records = self.aggregation_collection.find_one({"uid": "overall"})

            if not overall_records:
                logger.warning("No overall stats found")
                return self._empty_response("overall")
            return self._format_aggregation("overall", overall_records)
        
        except Exception as e:
            logger.error(f"Error retrieving overall stats: {e}")
            raise


    def get_daily_stats(self, date: Optional[str] = None) -> dict:
        """Get daily stats for a specific date or today"""
        try:
            if not date:
                date = self.now.strftime("%Y-%m-%d")

            daily_records = self.aggregation_collection.find_one({"uid": f"daily_{date}"})

            if not daily_records:
                logger.warning(f"No daily stats found for date: {date}")
                return self._empty_response("daily", date=date)
            return self._format_aggregation("daily", daily_records, date=date)
        
        except Exception as e:
            logger.error(f"Error retrieving daily stats for date '{date}': {e}")
            raise


    def get_monthly_stats(self, month: Optional[str] = None) -> dict:
        """Get monthly stats for a secific month or current month"""
        try:
            if not month:
                month =  self.now.strftime("%Y-%m")

            monthly_records = self.aggregation_collection.find_one({"uid": f"monthly_{month}"})

            if not monthly_records:
                logger.warning(f"No monthly stats found for month: {month}")
                return self._empty_response("monthly", month=month)
            return self._format_aggregation("monthly", monthly_records, month=month)
        
        except Exception as e:
            logger.error(f"Error retrieving monthly stats for month {month}: {e}")
            raise


    def get_range_stats(self, start_date: Optional[str] = None, end_date: Optional[str] = None, days: Optional[int] = None ) -> dict:
        """Get stats for a specifi date range or last x days"""
        try:
            if days:
                end_date = self.now.strftime("%Y-%m-%d")
                start_date = (self.now - timedelta(days=days)).strftime("%Y-%m-%d")
            elif not start_date or not end_date:
                raise ValueError("Provide either start_date and end_date or days")

            range_records = list(self.aggregation_collection.find({
                "agg_type":"daily",
                "agg_date":{
                    "$gte":start_date,
                    "$lte":end_date
                }
            }).sort("agg_date", 1))

            if not range_records:
                logger.warning(f"No stats found for this range from {start_date} to {end_date}")
                return self._empty_response("range", start_date=start_date, end_date=end_date)
            
            return{
            "scope": "range",
            "total_cost": sum(r.get("total_cost",0) for r in range_records),
            "total_input_tokens": sum(r.get('total_input_tokens', 0) for r in range_records),
            "total_output_tokens": sum(r.get('total_output_tokens', 0) for r in range_records),
            "document_count": sum(r.get('document_count', 0) for r in range_records),
            "daily_breakdown": [
                {
                    "date": r.get("agg_date"),
                    "cost": r.get("total_cost", 0),
                    "input_tokens": r.get("total_input_tokens", 0),
                    "output_tokens": r.get("total_output_tokens", 0),
                    "document_count": r.get("document_count", 0)
                }
                for r in range_records
            ]
            
            }
        
        except Exception as e:
            logger.error(f"Error retrieving range stats for {start_date} to {end_date}: {e}")
            raise


    def get_document_stats(self, document_id: Optional[int] = None) -> dict:
        """Get stats for a specific document by id"""
        try:
            if not document_id:
                raise ValueError("document_id is required")
            
            document_record = self.llm_cost_collection.find_one(
                {"uid": document_id},
                {"llm_cost_tracking": 1, "uid": 1}
            )

            if not document_record:
                raise ValueError(f"Document {document_id} not found")
            
            if "llm_cost_tracking" not in document_record:
                raise ValueError(f"Document {document_id} has no cost tracking data")
            
            llm_cost_tracking_data = document_record["llm_cost_tracking"]

            return{
                "scope": "document",
                "document_id": document_record.get("uid"),
                "total_cost" : llm_cost_tracking_data.get("total_cost",0),
                "total_input_tokens": llm_cost_tracking_data.get("total_input_tokens", 0),
                "total_output_tokens": llm_cost_tracking_data.get("total_output_tokens", 0),
                "tracked_at": llm_cost_tracking_data.get("tracked_at"),
                "usage_records": llm_cost_tracking_data.get("usage_records", [])
               }
        
        except Exception as e:
            logger.error(f"Error retrieving document stats for document_id {document_id}: {e}")
            raise


    def get_stats(self, scope: str, **kwargs) -> dict:
        """Orchestrator function to get stats according to scope"""
        if scope not in self.VALID_SCOPES:
            raise ValueError(f"Invalid scope: {scope}")
        
        try:
            match scope:
                case "overall":
                    return self.get_overall_stats()
                
                case "daily":
                    return self.get_daily_stats(date=kwargs.get("date"))
                
                case "monthly":
                    return self.get_monthly_stats(month=kwargs.get("month"))
                
                case "range":
                    return self.get_range_stats(start_date=kwargs.get("start_date"), end_date=kwargs.get("end_date"), days=kwargs.get("days"))
                
                case "document":
                    return self.get_document_stats(document_id=kwargs.get("document_id"))
        
        except Exception as e:
            logger.error(f"Error retrieving stats for scope '{scope}': {e}")
            raise