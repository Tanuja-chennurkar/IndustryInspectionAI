"""
FastAPI REST API Router Definitions.
Implements endpoints: inspect, history, statistics, defects, models, health, camera stream.
"""

from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import List, Optional
from datetime import datetime

from backend.app.schemas.inspection import (
    InspectionResponseSchema,
    InspectionListItemSchema,
    StatisticsResponseSchema,
    DefectItemSchema,
    ModelMetadataItemSchema,
    HealthResponseSchema
)
from backend.app.services.inspection_service import inspection_service
from backend.app.services.camera_service import camera_service
from backend.app.database.repository import repository
from ml.datasets.visa_dataset import VISA_CATEGORIES

api_router = APIRouter(prefix="/api", tags=["Industrial Inspection APIs"])

@api_router.post(
    "/inspect",
    response_model=InspectionResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Upload image and execute real-time anomaly detection & localization"
)
async def inspect_image(
    file: UploadFile = File(..., description="Industrial product image file (JPEG/PNG)"),
    category: str = Form(..., description="Required VisA product category"),
    threshold: Optional[float] = Form(None, description="Optional custom anomaly threshold override")
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid image (JPEG/PNG)"
        )

    if category not in VISA_CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported category '{category}'. Choose one of: {', '.join(VISA_CATEGORIES)}"
        )

    try:
        contents = await file.read()
        record = inspection_service.process_inspection(
            image_bytes=contents,
            category=category,
            custom_threshold=threshold
        )
        return record
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(e)}")

@api_router.get(
    "/inspections",
    response_model=List[InspectionListItemSchema],
    summary="Retrieve paginated inspection history"
)
def get_inspections(
    category: Optional[str] = Query(None, description="Filter by category ('candle', 'pcb1', etc.)"),
    status: Optional[str] = Query(None, description="Filter by status ('NORMAL' or 'ANOMALY')"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return")
):
    records = repository.get_all_inspections(category=category, status=status, limit=limit)
    return records

@api_router.get(
    "/inspections/{inspection_id}",
    response_model=InspectionResponseSchema,
    summary="Get detailed inspection record by ID"
)
def get_inspection_by_id(inspection_id: str):
    record = repository.get_inspection_by_id(inspection_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection ID '{inspection_id}' not found"
        )
    return record

@api_router.get(
    "/statistics",
    response_model=StatisticsResponseSchema,
    summary="Get overall inspection statistics and category metrics"
)
def get_statistics():
    return repository.get_statistics()

@api_router.get(
    "/defects",
    response_model=List[DefectItemSchema],
    summary="List recent defective inspection records"
)
def get_defects(limit: int = Query(50, ge=1, le=500)):
    defects = repository.get_defects(limit=limit)
    res = []
    for d in defects:
        res.append({
            "inspection_id": d.get("inspection_id", ""),
            "category": d.get("category", ""),
            "anomaly_score": float(d.get("anomaly_score", 0.0)),
            "defect_area_percentage": float(d.get("defect_area_percentage", d.get("defect_area", 0.0))),
            "defect_regions_count": int(d.get("defect_regions_count", len(d.get("defect_regions", d.get("defect_location", []))))),
            "timestamp": str(d.get("timestamp", "")),
            "image_url": str(d.get("image_url", d.get("image_reference", "")))
        })
    return res

@api_router.get(
    "/models",
    response_model=List[ModelMetadataItemSchema],
    summary="Get status and metadata for trained category anomaly models"
)
def get_models():
    models_metadata = []
    detector = inspection_service.detector
    for cat in VISA_CATEGORIES:
        thresh = detector.category_manager.category_thresholds.get(cat, 0.35)
        models_metadata.append({
            "category": cat,
            "version": detector.category_manager.model_version,
            "threshold": round(thresh, 4),
            "input_shape": list(detector.image_size) + [3],
            "weights_file": f"saved_models/{cat}/model.weights.h5"
        })
    return models_metadata

@api_router.get(
    "/health",
    response_model=HealthResponseSchema,
    summary="Service healthcheck endpoint"
)
def get_health():
    return {
        "status": "HEALTHY",
        "service": "Industrial Visual Inspection AI Backend",
        "model_loaded": True,
        "num_categories": len(VISA_CATEGORIES),
        "timestamp": datetime.now().isoformat()
    }

# ==================== Real-Time Camera Endpoints ====================

@api_router.get("/camera/stream", summary="Live MJPEG Camera Stream Feed")
def get_camera_stream():
    return StreamingResponse(
        camera_service.generate_mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@api_router.post("/camera/start", summary="Start camera acquisition stream")
def start_camera():
    success = camera_service.start_camera()
    return {"status": "SUCCESS" if success else "ERROR", "camera_active": camera_service.is_running}

@api_router.post("/camera/stop", summary="Stop camera acquisition stream")
def stop_camera():
    success = camera_service.stop_camera()
    return {"status": "SUCCESS" if success else "ERROR", "camera_active": camera_service.is_running}

@api_router.get("/camera/status", summary="Get real-time camera status")
def get_camera_status():
    return camera_service.get_status()
