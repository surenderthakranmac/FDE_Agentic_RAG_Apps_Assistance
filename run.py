import os
import sys
from pathlib import Path
import uvicorn

# 1. Programmatically capture the absolute path to this project folder
APP_ROOT = str(Path(__file__).resolve().parent)

# 2. Force append it to python's system environments before initializing uvicorn
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)
os.environ["PYTHONPATH"] = APP_ROOT

if __name__ == "__main__":
    # Import your ingestion function locally inside the runner frame
    # from app.services.ingestion import run_native_ingestion

    # # Run the force reindex function natively
    # run_native_ingestion(force_reindex=True)
    from app.core.config import settings
    from app.core.logger_setup import logger

    logger.info(f"⚡ Launching production web server cluster for: {settings.app_name}")

    # Run the server loop natively using string routing syntax
    uvicorn.run("app.main:app", host="127.0.0.1", port=8001, reload=False)
