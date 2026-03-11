from backend.core.config import settings
from backend.database import collection
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)       # Update existing cost tracking in MongoDB (append to existing)
logger = logging.getLogger(__name__)

MODEL_PRICING = {
    "gemini-2.5-flash": {
        "input": 0.30,  
        "output": 2.50, 
    },
    "gemini-2.5-flash-lite": {
        "input": 0.10,  
        "output": 0.40, 
    },
    "gemini-2.0-flash": {
        "input": 0.30,
        "output": 1.20,
    },
    "mistral-ocr-latest": {
        "input": 0.00,  
        "output": 0.00,
    },
    "mistral-small-latest": {
        "input": 0.00, 
        "output": 0.00,
    }
}


class LLMCostTracker:
    """
    Cost tracker for LLM usage. Tracks input/output tokens and calculates costs.
    """
    
    def __init__(self, model_name: str = settings):
        self.model_name = model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.usage_records = []  # List to store individual usage records
    
    def calculate_llm_input_tokens(self, input_tokens: int) -> int:
        """Add LLM input tokens to the running total."""
        self.total_input_tokens += input_tokens
        return self.total_input_tokens

    def calculate_llm_output_tokens(self, output_tokens: int) -> int:
        """Add LLM output tokens to the running total."""
        self.total_output_tokens += output_tokens
        return self.total_output_tokens

    def _calculate_usage_cost(self, input_tokens: int, output_tokens: int, model_name: Optional[str] = None) -> float:
        """Calculate cost for a single LLM usage based on token counts."""
        model = model_name or self.model_name
        pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    def track_llm_usage(self, input_tokens: int, output_tokens: int, operation: str = "",model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Track a single LLM invocation. Call this after every LLM invoke.
        
        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
            operation: Description of the operation (e.g., "extract_vendor_details")
            model_name: Optional model name override
        
        Returns:
            Dict containing the usage details and cost for this invocation
        """
        model = model_name or self.model_name

        cost = self._calculate_usage_cost(input_tokens, output_tokens, model)

        self.calculate_llm_input_tokens(input_tokens)
        self.calculate_llm_output_tokens(output_tokens)
        self.total_cost += cost

        usage_record = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost
        }
        
        self.usage_records.append(usage_record)
        
        logger.info(f"LLM Usage tracked - Operation: {operation}, Model: {model}, "
                   f"Input: {input_tokens}, Output: {output_tokens}, Cost: ${cost:.6f}")
        
        return usage_record
    
    def track_fixed_cost(self, cost:float, operation: str, model_name: str) -> Dict[str, Any]:

        self.total_cost += cost

        usage_record = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "operation": operation,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": cost
        }

        self.usage_records.append(usage_record)
        logger.info(f"Fixed cost tracked - Operation: {operation}, Model: {model_name}, Cost: ${cost:.6f}")
        return usage_record

    def get_total_usage(self) -> Dict[str, Any]:
        """Get the total accumulated usage and cost."""
        return {
            "model": self.model_name,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            "num_invocations": len(self.usage_records),
            "usage_records": self.usage_records
        }
    
    async def save_to_mongodb(self, document_uid: int) -> bool:
        """
        Save the accumulated LLM cost data to MongoDB before mapping.
        
        Args:
            document_uid: The UID of the document to update
        
        Returns:
            True if save was successful, False otherwise
        """
        try:
            cost_data = {
                "llm_cost_tracking": {
                    "model": self.model_name,
                    "total_input_tokens": self.total_input_tokens,
                    "total_output_tokens": self.total_output_tokens,
                    "total_cost": self.total_cost,
                    "usage_records": self.usage_records,
                    "tracked_at": datetime.now().isoformat()
                }
            }
            
            result = await collection.update_one(
                {"uid": document_uid},
                {"$set": cost_data}
            )
            
            if result.modified_count > 0:
                logger.info(f"Successfully saved LLM cost data for document UID {document_uid}. "
                           f"Total cost: ${self.total_cost:.6f}")
                return True
            else:
                logger.warning(f"No document updated for UID {document_uid}. Document may not exist.")
                return False
                
        except Exception as e:
            logger.error(f"Error saving LLM cost data to MongoDB: {e}")
            return False
    
    def reset(self):
        """Reset all counters and records for a new tracking session."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.usage_records = []
        logger.info("LLM cost tracker reset")


def extract_langchain_usage(response) -> Dict[str, int]:
    """
    Extract token usage from a LangChain response object.
    Works with ChatGoogleGenerativeAI responses.
    
    Args:
        response: LangChain response object with usage_metadata
    
    Returns:
        Dict with input_tokens and output_tokens
    """
    try:
        usage_metadata = getattr(response, 'usage_metadata', None)

        # handle both dict and object formats
        if isinstance(usage_metadata, dict):
            input_tokens = usage_metadata.get("prompt_token_count", 0) or usage_metadata.get("input_tokens", 0)
            candidates_tokens = usage_metadata.get("candidates_token_count", 0) or usage_metadata.get("output_tokens", 0)
            thoughts_tokens = usage_metadata.get("thoughts_token_count", 0)
        else:            
            input_tokens = getattr(usage_metadata, 'prompt_token_count', 0) or getattr(usage_metadata, 'input_tokens', 0) or 0
            # For Gemini, output tokens = candidates_token_count + thoughts_token_count
            candidates_tokens = getattr(usage_metadata, 'candidates_token_count', 0) or getattr(usage_metadata, 'output_tokens', 0) or 0
            thoughts_tokens = getattr(usage_metadata, 'thoughts_token_count', 0) or 0
            
        candidates_tokens = candidates_tokens or 0    
        thoughts_tokens = thoughts_tokens or 0
        output_tokens = candidates_tokens + thoughts_tokens
        
        return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
        }  
        
    except Exception as e:
        logger.error(f"Error extracting usage from response: {e}")
        return {"input_tokens": 0, "output_tokens": 0}

cost_tracker = LLMCostTracker()
    