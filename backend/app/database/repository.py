"""
MongoDB & Hybrid Repository Layer for Inspection Records.
Manages persistent storage across collections:
- inspections
- products
- defects
- model_versions
"""

import os
from typing import List, Dict, Optional, Any
from datetime import datetime

from backend.app.database.connection import db_manager

class InspectionRepository:
    """
    Data Repository for managing MongoDB collections and fallback in-memory records.
    """
    def __init__(self):
        self.in_memory_records: List[Dict[str, Any]] = []
        self.counter: int = 10000
        # Initialize connection
        db_manager.connect()
        if db_manager.is_connected and db_manager.db is not None:
            latest_record = db_manager.db.inspections.find_one(
                {}, {"inspection_id": 1}, sort=[("_id", -1)]
            )
            latest_id = latest_record.get("inspection_id", "") if latest_record else ""
            if latest_id.startswith("INS-") and latest_id[4:].isdigit():
                self.counter = max(self.counter, int(latest_id[4:]))

    def generate_inspection_id(self) -> str:
        self.counter += 1
        return f"INS-{self.counter}"

    def save_inspection(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Saves inspection record into 'inspections' and 'defects' collections.
        """
        # Ensure mandatory VisA schema fields exist
        product_id = record.get("product_id", f"PROD-{(record.get('category', 'item')).upper()}-01")
        record["product_id"] = product_id

        doc = {
            "inspection_id": record["inspection_id"],
            "product_id": product_id,
            "category": record["category"],
            "category_confidence": record.get("category_confidence", 1.0),
            "timestamp": record.get("timestamp", datetime.now().isoformat()),
            "status": record["status"],
            "anomaly_score": float(record["anomaly_score"]),
            "threshold": float(record["threshold"]),
            "defect_type": "surface_defect" if record["status"] == "ANOMALY" else "none",
            "defect_area": float(record.get("defect_area_percentage", 0.0)),
            "defect_location": record.get("defect_regions", []),
            "model_version": record.get("model_version", "v1.0.0"),
            "processing_time": float(record.get("processing_time_ms", 0.0)),
            "image_reference": record.get("image_url", ""),
            "created_at": datetime.now()
        }

        # 1. MongoDB Persistence
        if db_manager.is_connected and db_manager.db is not None:
            try:
                db_manager.db.inspections.insert_one(doc.copy())

                # If anomaly, also insert into defects collection
                if record["status"] == "ANOMALY":
                    db_manager.db.defects.insert_one({
                        "defect_id": f"DEF-{record['inspection_id']}",
                        "inspection_id": record["inspection_id"],
                        "category": record["category"],
                        "anomaly_score": float(record.get("anomaly_score", 0.0)),
                        "defect_area": float(record.get("defect_area_percentage", 0.0)),
                        "defect_regions_count": len(record.get("defect_regions", [])),
                        "timestamp": record.get("timestamp", datetime.now().isoformat()),
                        "image_url": record.get("image_url", "")
                    })

                # Register in products collection
                db_manager.db.products.update_one(
                    {"product_id": product_id},
                    {"$set": {"category": record["category"], "updated_at": datetime.now()}},
                    upsert=True
                )
            except Exception as e:
                print(f"[MongoDB Insert Warning] {e}")

        # 2. In-Memory Cache Persistence
        self.in_memory_records.append(record)
        return record

    def _normalize_record(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures consistent API schema format regardless of whether source is MongoDB or in-memory.
        """
        if not doc:
            return doc
        normalized = doc.copy()
        if "_id" in normalized:
            del normalized["_id"]
        
        # Map defect_area -> defect_area_percentage
        if "defect_area_percentage" not in normalized and "defect_area" in normalized:
            normalized["defect_area_percentage"] = float(normalized["defect_area"])
        elif "defect_area_percentage" not in normalized:
            normalized["defect_area_percentage"] = 0.0

        # Map processing_time -> processing_time_ms
        if "processing_time_ms" not in normalized and "processing_time" in normalized:
            normalized["processing_time_ms"] = float(normalized["processing_time"])
        elif "processing_time_ms" not in normalized:
            normalized["processing_time_ms"] = 0.0

        # Map image_reference -> image_url
        if "image_url" not in normalized and "image_reference" in normalized:
            normalized["image_url"] = str(normalized["image_reference"])
        elif "image_url" not in normalized:
            normalized["image_url"] = ""

        # Map defect_location -> defect_regions
        if "defect_regions" not in normalized and "defect_location" in normalized:
            normalized["defect_regions"] = normalized["defect_location"]
        elif "defect_regions" not in normalized:
            normalized["defect_regions"] = []

        return normalized

    def get_all_inspections(
        self,
        category: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Queries inspections collection with filtering.
        """
        if db_manager.is_connected and db_manager.db is not None:
            try:
                query = {}
                if category and category != "all":
                    query["category"] = category
                if status and status != "all":
                    query["status"] = status

                cursor = db_manager.db.inspections.find(query, {"_id": 0}).sort("timestamp", -1).limit(limit)
                docs = [self._normalize_record(d) for d in cursor]
                if docs:
                    return docs
            except Exception as e:
                print(f"[MongoDB Query Warning] {e}")

        # Fallback to cache
        results = [self._normalize_record(r) for r in self.in_memory_records]
        if category and category != "all":
            results = [r for r in results if r.get("category") == category]
        if status and status != "all":
            results = [r for r in results if r.get("status") == status]
        return sorted(results, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]

    def get_inspection_by_id(self, inspection_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetches inspection document by inspection_id.
        """
        if db_manager.is_connected and db_manager.db is not None:
            try:
                doc = db_manager.db.inspections.find_one({"inspection_id": inspection_id}, {"_id": 0})
                if doc:
                    return self._normalize_record(doc)
            except Exception as e:
                print(f"[MongoDB ID Query Warning] {e}")

        for r in self.in_memory_records:
            if r.get("inspection_id") == inspection_id:
                return self._normalize_record(r)
        return None

    def get_defects(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Queries defects collection.
        """
        if db_manager.is_connected and db_manager.db is not None:
            try:
                cursor = db_manager.db.defects.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
                docs = list(cursor)
                if docs:
                    return docs
            except Exception as e:
                print(f"[MongoDB Defects Query Warning] {e}")

        anomalies = [r for r in self.in_memory_records if r.get("status") == "ANOMALY"]
        return sorted(anomalies, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculates aggregate statistics across inspections collection.
        """
        inspections = self.get_all_inspections(limit=10000)
        total = len(inspections)
        normal = len([r for r in inspections if r.get("status") == "NORMAL"])
        anomaly = len([r for r in inspections if r.get("status") == "ANOMALY"])
        defect_rate = (anomaly / total * 100.0) if total > 0 else 0.0
        avg_score = (sum(float(r.get("anomaly_score", 0.0)) for r in inspections) / total) if total > 0 else 0.0

        cat_stats = {}
        for r in inspections:
            cat = r.get("category", "unknown")
            if cat not in cat_stats:
                cat_stats[cat] = {"total": 0, "normal": 0, "anomaly": 0}
            cat_stats[cat]["total"] += 1
            if r.get("status") == "NORMAL":
                cat_stats[cat]["normal"] += 1
            else:
                cat_stats[cat]["anomaly"] += 1

        return {
            "total_inspections": total,
            "normal_count": normal,
            "anomaly_count": anomaly,
            "defect_rate_percentage": round(defect_rate, 2),
            "average_anomaly_score": round(avg_score, 4),
            "category_breakdown": cat_stats
        }

# Global singleton repository instance
repository = InspectionRepository()
