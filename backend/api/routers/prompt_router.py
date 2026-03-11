from fastapi import APIRouter
from fastapi.responses import JSONResponse
from backend.database import collection

router = APIRouter(prefix="/prompts", tags=["Prompt Management"])

@router.put("/update-prompt")
async def update_prompt(doc_id: int, prompt: str):
    try:
        document = await collection.find_one({
            "uid": doc_id,
            "prompt_type": "user_given_prompt"
        })
        
        if not document:
            return JSONResponse(
                status_code=404,
                content={
                    "message": "Document not found or is not a user-given prompt document"
                }
            )
        
        result = await collection.update_one(
            {"uid": doc_id},
            {"$set": {"prompt": prompt}}
        )
        
        if result.modified_count > 0:
            return JSONResponse(
                status_code=200,
                content={"message": f"Prompt updated successfully for document {doc_id}"}
            )
        else:
            return JSONResponse(
                status_code=400,
                content={"message": "No changes were made to the document"}
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error updating prompt: {str(e)}"}
        )    

@router.get("/default-prompts")
async def get_default_prompts():
    try:
        default_prompt = list(await collection.find({"default_type": {"$exists": True}},{"_id": 0}))
        if not default_prompt:
            return {"message": "No default prompts found"}
        else:
            return default_prompt
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error fetching default prompts: {str(e)}"}
        ) 
