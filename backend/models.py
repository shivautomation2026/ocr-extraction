from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any, Literal, Optional

Literal
class OCRResponse(BaseModel):
    status: str
    message: str
    content: Dict[str, Any]
    extracted_text: str

class CostStatsRequest(BaseModel):
    model_config = {"extra": "forbid"}
    scope: Literal["overall", "daily", "monthly", "range", "document"] = Field("overall", description="Type of statistics to fetch")
    date: Optional[str] = Field(None, description="Date for daily scope (YYYY-MM-DD)", pattern=r"^\d{4}-\d{2}-\d{2}$")
    month: Optional[str] = Field(None, description="Month for monthly scope (YYYY-MM)", pattern=r"^\d{4}-\d{2}$")
    start_date: Optional[str] = Field(None, description="Start date for range scope", pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: Optional[str] = Field(None, description="End date for range scope", pattern=r"^\d{4}-\d{2}-\d{2}$")
    days: Optional[int] = Field(None, description="Number of days for range scope", gt=1, lt= 366)
    document_id: Optional[int] = Field(None, description="Document ID for document scope")

    @model_validator(mode="after")
    def validate_scope_fields(self)-> "CostStatsRequest":
        """to make sure necessary params are present for scope"""
        match self.scope:
            case "daily":
                if not self.date:
                    raise ValueError("date is required for daily scope")
                
            case "monthly":
                if not self.month:
                    raise ValueError("month is required for monthly scope")
            
            case "range":
                if not self.days and not (self.start_date and self.end_date):
                    raise ValueError("days or start_date and end_date are required for range scope")
            
            case "document":
                if not self.document_id:
                    raise ValueError("document_id is required for document scope")