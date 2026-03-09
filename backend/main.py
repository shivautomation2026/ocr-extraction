from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from .api.routers import extraction_router, prompt_router, mapping_router, sap_invoice_router, cost_tracker_router, review_router
from .database import check_collection
from contextlib import asynccontextmanager
recent_filename = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.collection = await check_collection()
    
    yield
    

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return HTMLResponse("""
    <h1>Welcome to the OCR Extraction and Classification API</h1>
    """)
    
app.include_router(extraction_router.router)
app.include_router(review_router.router)
app.include_router(prompt_router.router)
app.include_router(mapping_router.router)
app.include_router(sap_invoice_router.router)
app.include_router(cost_tracker_router.router)
