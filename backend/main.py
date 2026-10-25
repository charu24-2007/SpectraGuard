"""
SpectraGuard FastAPI Application Entry Point.
Production-configured passive RF spectrum intelligence platform.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.api.routes import router
from backend.database.database import init_db
from backend.services.model_service import model_service
from backend.services.dataset_service import dataset_service

# Logging setup
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("spectraguard.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting SpectraGuard Application...")
    logger.info("Initializing database schema and indexes...")
    init_db()

    logger.info("Loading evaluated model checkpoints...")
    try:
        model_service.load()
    except Exception as e:
        logger.error(f"Error loading models at startup: {e}")

    try:
        dataset_service._open()
    except Exception as e:
        logger.warning(f"Note: Dataset mount deferred or unavailable at startup: {e}")

    logger.info("SpectraGuard startup complete.")
    yield
    # Shutdown
    logger.info("SpectraGuard shutting down...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configured CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
