"""
Anomaly Heatmap & Localization Visualization Module.
Generates JET colormap overlays, status badges, defect bounding boxes,
and multi-panel side-by-side diagnostic visualizations.
"""

import os
import cv2
import numpy as np
from typing import Dict, Optional, Tuple

class HeatmapVisualizer:
    """
    Visualizer for rendering industrial anomaly heatmaps and overlays.
    """
    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha

    def generate_heatmap_overlay(
        self,
        raw_rgb: np.ndarray,
        anomaly_map: np.ndarray,
        defect_regions: Optional[list] = None,
        status: str = "NORMAL",
        anomaly_score: float = 0.0,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Creates an alpha-blended JET heatmap overlay with defect bounding boxes and status badge.
        """
        # Normalize anomaly map to [0, 255] uint8
        norm_map = np.clip(anomaly_map / max(1e-5, np.max(anomaly_map)), 0, 1)
        heatmap_gray = (norm_map * 255.0).astype(np.uint8)
        
        # Apply JET colormap
        heatmap_bgr = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Alpha blend original image with heatmap
        overlay_rgb = cv2.addWeighted(raw_rgb, 1.0 - self.alpha, heatmap_rgb, self.alpha, 0)

        # Draw defect bounding boxes if status is ANOMALY
        if status == "ANOMALY" and defect_regions:
            for region in defect_regions:
                x, y, w, h = region["bbox"]
                cv2.rectangle(overlay_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red box
                # Draw centroid mark
                cx, cy = region["centroid"]
                cv2.circle(overlay_rgb, (cx, cy), 3, (255, 255, 0), -1)

        # Draw status badge on top left corner
        badge_color = (0, 200, 0) if status == "NORMAL" else (220, 0, 0)
        cv2.rectangle(overlay_rgb, (10, 10), (220, 45), (0, 0, 0), -1)  # background pill
        cv2.rectangle(overlay_rgb, (10, 10), (220, 45), badge_color, 2)
        
        text = f"{status} ({anomaly_score:.2f}/{threshold:.2f})"
        cv2.putText(overlay_rgb, text, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return overlay_rgb

    def create_multipanel_visualization(self, inspection_result: Dict) -> np.ndarray:
        """
        Generates a 4-panel side-by-side composite diagnostic image:
        [1. Original Image | 2. Reconstructed | 3. Raw Anomaly Map | 4. Heatmap Overlay]
        """
        raw_rgb = inspection_result["raw_image"]
        recon_rgb = inspection_result["reconstructed_image"]
        anomaly_map = inspection_result["anomaly_map"]
        status = inspection_result["status"]
        score = inspection_result["anomaly_score"]
        threshold = inspection_result["threshold"]
        defect_regions = inspection_result.get("defect_regions", [])

        # 1. Heatmap overlay
        overlay = self.generate_heatmap_overlay(
            raw_rgb=raw_rgb,
            anomaly_map=anomaly_map,
            defect_regions=defect_regions,
            status=status,
            anomaly_score=score,
            threshold=threshold
        )

        # 2. Raw Anomaly Map (Gray to RGB)
        norm_map = np.clip(anomaly_map / max(1e-5, np.max(anomaly_map)), 0, 1)
        map_rgb = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        map_rgb = cv2.cvtColor(map_rgb, cv2.COLOR_BGR2RGB)

        # Annotate panel titles
        def annotate_panel(img, title):
            res = img.copy()
            cv2.rectangle(res, (0, 0), (res.shape[1], 25), (20, 20, 20), -1)
            cv2.putText(res, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
            return res

        p1 = annotate_panel(raw_rgb, "1. Input Image")
        p2 = annotate_panel(recon_rgb, "2. Normal Reconstruction")
        p3 = annotate_panel(map_rgb, "3. Anomaly Error Map")
        p4 = annotate_panel(overlay, f"4. Defect Heatmap ({status})")

        # Horizontal concatenation
        composite = np.hstack([p1, p2, p3, p4])
        return composite
