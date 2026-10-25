# import os
# import sys
# from pathlib import Path
# import uvicorn

# # ============================================================
# # 📁 GLOBAL PATH INSULATION (Fixes Subprocess ModuleNotFoundError)
# # ============================================================
# # Resolve the absolute parent directory containing the 'app' module folder
# ROOT_DIR = str(Path(__file__).resolve().parent)

# # Inject path to the global environment and python runtime lookup table
# if ROOT_DIR not in sys.path:
#     sys.path.insert(0, ROOT_DIR)
# os.environ["PYTHONPATH"] = ROOT_DIR + (
#     os.pathsep + os.environ.get("PYTHONPATH", "")
#     if os.environ.get("PYTHONPATH")
#     else ""
# )

# from app.core.config import settings
# from app.core.logger_setup import logger

# if __name__ == "__main__":
#     logger.info(
#         f"⚡ Launching production-hardened web cluster server layer for: {settings.app_name}"
#     )

#     # Executes the FastAPI framework engine over the local loopback interface
#     uvicorn.run("app.main:app", host="127.0.0.1", port=8001, reload=True)
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
    from app.core.config import settings
    from app.core.logger_setup import logger

    logger.info(f"⚡ Launching production web server cluster for: {settings.app_name}")

    # Run the server loop natively using string routing syntax
    uvicorn.run("app.main:app", host="127.0.0.1", port=8001, reload=False)
