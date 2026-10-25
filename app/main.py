import os
import sys
from pathlib import Path

# ============================================================
# 📁 SYSTEM PATH INSULATION (Fixes ModuleNotFoundError)
# ============================================================
ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.core.logger_setup import logger
from app.api.routes import router as api_router
from app.services.ingestion import run_native_ingestion

# 1. Instantiate the central production FastAPI application instance
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Local Private Hardened Forward-Deployed Agentic RAG Copilot",
)

# 2. Configure Cross-Origin Resource Sharing (CORS) security context rules
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Mount static web assets and interactive templates directories safely
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the production template rendering engine mapping path parameters
templates = Jinja2Templates(directory="templates")

# 4. Include the complete sub-package API routing structure matrix
app.include_router(api_router)


# ============================================================
# 🖥️ FRONTEND WEB VIEW DASHBOARD GATEWAY (Fixes 404 Error)
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard_interface(request: Request):
    """Renders the employee-facing IT support chat view dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.on_event("startup")
async def startup_event():
    """Fires automatically when the Uvicorn web server mounts into memory."""
    logger.info(
        "Initializing Enterprise IT Support Agentic Copilot system process layer..."
    )
    try:
        # Pre-seed internal database arrays using your native PDF extractor loops on boot
        run_native_ingestion()
        logger.info(
            "Application layer baseline synchronization complete. Web service is live."
        )
    except Exception as e:
        logger.critical(f"Critical application boot sequence error caught: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Launch local serving framework over loopback network parameters
    uvicorn.run("app.main:app", host="127.0.0.1", port=8001, reload=True)
