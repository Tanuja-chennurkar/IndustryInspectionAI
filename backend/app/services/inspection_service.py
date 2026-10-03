"""
Inspection Business Logic Service.
Orchestrates image decoding, ML inference execution, heatmap image saving,
and repository record creation.
"""

import os
import sys
import time
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Optional

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from ml.inference.detector import AnomalyDetector
from ml.visualization.heatmap import HeatmapVisualizer
from backend.app.database.repository import repository

class InspectionService:
    """
    Service layer for industrial visual inspection.
    """
    def __init__(self, static_dir: str = "backend/static"):
        self.static_dir = os.path.abspath(static_dir)
        self.heatmaps_dir = os.path.join(self.static_dir, "heatmaps")
        os.makedirs(self.heatmaps_dir, exist_ok=True)
        
        self.detector = AnomalyDetector(saved_models_dir="ml/saved_models")
        self.visualizer = HeatmapVisualizer()

    def process_inspection(
        self,
        image_bytes: bytes,
        category: Optional[str] = None,
        custom_threshold: Optional[float] = None
    ) -> Dict:
        """
        Executes complete inspection workflow from raw uploaded bytes.
        """
        start_time = time.time()

        # Decode image bytes to OpenCV BGR array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise ValueError("Invalid image file format or corrupted bytes")

        # Execute ML Inference & Category Auto-Routing directly on raw image array
        result = self.detector.inspect_image(
            image_input=img_bgr,
            category=category,
            custom_threshold=custom_threshold,
            is_bgr=True
        )

        processing_time_ms = float((time.time() - start_time) * 1000.0)
        inspection_id = repository.generate_inspection_id()

        # Generate composite heatmap overlay image
        comp_img = self.visualizer.create_multipanel_visualization(result)
        filename = f"{inspection_id}_heatmap.png"
        file_path = os.path.join(self.heatmaps_dir, filename)
        
        # Save to disk as BGR
        cv2.imwrite(file_path, cv2.cvtColor(comp_img, cv2.COLOR_RGB2BGR))

        image_url = f"/static/heatmaps/{filename}"
        timestamp_str = datetime.now().isoformat()

        record = {
            "inspection_id": inspection_id,
            "category": result["category"],
            "category_confidence": result["category_confidence"],
            "status": result["status"],
            "anomaly_score": result["anomaly_score"],
            "threshold": result["threshold"],
            "defect_area_percentage": result["defect_area_percentage"],
            "defect_regions": result["defect_regions"],
            "model_version": result["model_version"],
            "processing_time_ms": round(processing_time_ms, 2),
            "timestamp": timestamp_str,
            "image_url": image_url
        }

        # Save to repository
        repository.save_inspection(record)
        return record

# Global service instance
inspection_service = InspectionService()
