"""
MongoDB Connection Lifecycle and Index Manager.
Handles database connections, collection initialization, and index creation.
"""

import os
import pymongo
from pymongo import MongoClient
from typing import Optional

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "industry_inspection_db")

class MongoDBManager:
    """
    MongoDB Connection and Index Management Class.
    """
    def __init__(self, uri: str = MONGODB_URI, db_name: str = MONGODB_DB_NAME):
        self.uri = uri
        self.db_name = db_name
        self.client: Optional[MongoClient] = None
        self.db = None
        self.is_connected = False

    def connect(self) -> bool:
        """
        Establishes connection to MongoDB instance.
        """
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=2000)
            # Check server ping
            self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            self.is_connected = True
            self.create_indexes()
            print(f"[MongoDB] Successfully connected to {self.uri} -> DB: {self.db_name}")
            return True
        except Exception as e:
            self.is_connected = False
            print(f"[MongoDB Notice] Could not connect to MongoDB server ({e}). Running in fallback mode.")
            return False

    def create_indexes(self):
        """
        Creates collection indexes for fast query performance.
        """
        if not self.is_connected or self.db is None:
            return

        try:
            # 1. inspections collection indexes
            self.db.inspections.create_index("inspection_id", unique=True)
            self.db.inspections.create_index("category")
            self.db.inspections.create_index("status")
            self.db.inspections.create_index([("timestamp", pymongo.DESCENDING)])

            # 2. products collection indexes
            self.db.products.create_index("product_id", unique=True)
            self.db.products.create_index("category")

            # 3. defects collection indexes
            self.db.defects.create_index("inspection_id")
            self.db.defects.create_index("category")

            # 4. model_versions collection indexes
            self.db.model_versions.create_index("category", unique=True)

            print("[MongoDB] Collection indexes verified successfully.")
        except Exception as e:
            print(f"[MongoDB Warning] Index creation notice: {e}")

# Global MongoDB manager instance
db_manager = MongoDBManager()
