"""
Pydantic Schemas for FastAPI Endpoints.
Defines strict request/response data contracts.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class DefectRegionSchema(BaseModel):
    bbox: List[int] = Field(..., description="[x, y, width, height] bounding box")
    area_pixels: float = Field(..., description="Defect area in pixels")
    centroid: List[int] = Field(..., description="[cx, cy] centroid coordinate")

class InspectionResponseSchema(BaseModel):
    inspection_id: str = Field(..., example="INS-10001")
    category: str = Field(..., example="candle")
    category_confidence: float = Field(1.0, example=0.985)
    status: str = Field(..., example="ANOMALY")
    anomaly_score: float = Field(..., example=0.91)
    threshold: float = Field(..., example=0.65)
    defect_area_percentage: float = Field(..., example=3.8)
    defect_regions: List[DefectRegionSchema] = Field(default_factory=list)
    model_version: str = Field(..., example="v1.0.0")
    processing_time_ms: float = Field(..., example=84.2)
    timestamp: str = Field(..., example="2026-09-12T19:50:00")
    image_url: str = Field(..., example="/static/heatmaps/INS-10001_heatmap.png")

class InspectionListItemSchema(BaseModel):
    inspection_id: str
    category: str
    status: str
    anomaly_score: float
    threshold: float
    defect_area_percentage: float
    timestamp: str
    image_url: str

class StatisticsResponseSchema(BaseModel):
    total_inspections: int
    normal_count: int
    anomaly_count: int
    defect_rate_percentage: float
    average_anomaly_score: float
    category_breakdown: Dict[str, Dict[str, Any]]

class DefectItemSchema(BaseModel):
    inspection_id: str
    category: str
    anomaly_score: float
    defect_area_percentage: float
    defect_regions_count: int
    timestamp: str
    image_url: str

class ModelMetadataItemSchema(BaseModel):
    category: str
    version: str
    threshold: float
    input_shape: List[int]
    weights_file: str

class HealthResponseSchema(BaseModel):
    status: str
    service: str
    model_loaded: bool
    num_categories: int
    timestamp: str
