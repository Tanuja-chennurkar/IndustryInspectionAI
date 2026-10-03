"""
Main FastAPI Web Application Entry Point.
Configures ASGI server, CORS middleware, static file serving, and REST routes.
"""

import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app.api.router import api_router

app = FastAPI(
    title="Self-Supervised Anomaly Detection for Industrial Visual Inspection API",
    description="Production REST backend powered by TensorFlow/Keras, VisA dataset, OpenCV, and MongoDB.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for React dashboard frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directories exist and mount static file server
static_dir = os.path.abspath("backend/static")
os.makedirs(os.path.join(static_dir, "heatmaps"), exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Register API routes
app.include_router(api_router)

@app.get("/", tags=["Root"])
def root():
    return {
        "title": "Industrial Visual Inspection AI System API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/api/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
