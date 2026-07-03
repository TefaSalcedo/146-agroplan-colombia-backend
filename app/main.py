from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.routers import municipalities

settings = get_settings()

app = FastAPI(
    title="AgroPlan Colombia Backend",
    description="Backend API for AgroPlan Colombia - Agricultural Intelligence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "AgroPlan Colombia Backend API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "models_loaded": False,
    }

# Include routers
app.include_router(municipalities.router, prefix=settings.api_v1_prefix)
