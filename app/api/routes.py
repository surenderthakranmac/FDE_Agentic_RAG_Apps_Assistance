import os
import time
import shutil
from fastapi import APIRouter, HTTPException, Header, UploadFile, File
from pydantic import BaseModel, Field
from app.rag.workflow import ask

from app.core.config import settings
from app.core.logger_setup import logger
from app.services.ingestion import run_native_ingestion, ALLOWED_OFFICE_EXTENSIONS

router = APIRouter(prefix="/api")


class ChatRequest(BaseModel):
    question: str = Field(
        ..., description="The unedited support query submitted by the employee."
    )

# 2. Append the missing chat route right here:
@router.post("/chat")
def chat(payload: ChatRequest):
    # Track when the generation begins
    start_time = time.perf_counter()
    try:
        # Extract whichever query parameter key your frontend successfully populated
        user_query = payload.question
        
        if not user_query:
            raise HTTPException(status_code=400, detail="Missing required string payload: 'message' or 'question'.")

        logger.info(f"📬 Received message payload: '{user_query}'")
        
        # Invoke your RAG agent workflow graph framework instance
        result = ask(user_query)
        
        # Ensure robust return lookups to prevent KeyError blocks if keys are missing
       # Calculate full processing time in milliseconds
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        return {
            "answer": result.get("answer", "No answer compiled by agent node."),
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result["current_query"],
            
            # ──► ADD THESE KEYS TO FIX THE FRONTEND "UNDEFINED" BLOCKS:
            "latency_ms": elapsed_ms,
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0)
        }

    except Exception as e:
        logger.error(f"Error handling chat execution sequence: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal agent orchestration error: {str(e)}"
        )



@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...), x_admin_key: str = Header(..., alias="X-Admin-Key")
):
    """
    Administrative Ingestion Gateway Endpoint.
    Enforces security header authorization checks and type validation constraints.
    """
    # 1. Enforce access token authorization checks
    if x_admin_key != settings.admin_api_key:
        logger.warning(
            "Rejected unauthorized ingestion deployment attempt via matching header token failure."
        )
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Invalid administrative access key token parameters.",
        )

    # 2. 🛡️ FILE EXTENSION VALIDATION FIREWALL GATEWAY
    filename_lower = file.filename.lower()
    if not filename_lower.endswith(ALLOWED_OFFICE_EXTENSIONS):
        logger.error(
            f"Ingestion validation rejection flag fired. Illegal format match attempted: '{file.filename}'"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Validation Failure: Format not whitelisted. System only accepts: {', '.join(ALLOWED_OFFICE_EXTENSIONS)}",
        )

    try:
        # Resolve target storage directories dynamically
        os.makedirs(settings.lectures_folder, exist_ok=True)
        destination_path = os.path.join(settings.lectures_folder, file.filename)

        # Stream chunks directly onto local system volumes
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(
            f"New structural asset binary saved: '{file.filename}'. Forcing indexing update sweep..."
        )

        # Trigger the optimized incremental processing pipeline execution cycle
        run_native_ingestion()

        return {
            "message": f"Asset '{file.filename}' processed and indexed successfully.",
            "status": "synchronized",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"In-flight administrative data data ingestion failure exception: {str(e)}"
        )
        raise HTTPException(
            status_code=500, detail=f"Data Sync Failure Anomaly: {str(e)}"
        )
