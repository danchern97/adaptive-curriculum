# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Local storage implementation that mimics MongoDB interface for searchformer.
This allows the codebase to work without requiring a MongoDB server.
"""

import json
import os
import pickle
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


class LocalStorage:
    """Local file-based storage that mimics MongoDB behavior."""
    
    def __init__(self, base_path: str = "data"):
        """
        Initialize local storage.
        
        Args:
            base_path: Base directory for all data storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Create subdirectories for different data types
        (self.base_path / "tokenSeqDB").mkdir(exist_ok=True)
        (self.base_path / "trainDB").mkdir(exist_ok=True)
        (self.base_path / "ckptDB").mkdir(exist_ok=True)
        (self.base_path / "rolloutDB").mkdir(exist_ok=True)
    
    def get_collection_path(self, db_name: str, collection_name: str) -> Path:
        """Get the path for a collection directory."""
        path = self.base_path / db_name / collection_name
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_gridfs_path(self, db_name: str) -> Path:
        """Get the path for GridFS files."""
        path = self.base_path / db_name / "_gridfs"
        path.mkdir(parents=True, exist_ok=True)
        return path


class LocalCollection:
    """Local file-based collection that stores all documents in a single JSON file."""
    
    def __init__(self, storage: LocalStorage, db_name: str, collection_name: str):
        self.storage = storage
        self.db_name = db_name
        self.collection_name = collection_name
        self.collection_file = storage.base_path / db_name / f"{collection_name}.json"
        self.collection_file.parent.mkdir(parents=True, exist_ok=True)
        
        # For read operations, we'll load documents on demand
        self._documents_cache = None
        self._cache_dirty = False
    
    def _load_documents(self):
        """Load all documents from the collection file."""
        if self._documents_cache is not None and not self._cache_dirty:
            return self._documents_cache
            
        if self.collection_file.exists():
            try:
                with open(self.collection_file, 'r') as f:
                    self._documents_cache = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self._documents_cache = {}
        else:
            self._documents_cache = {}
        
        self._cache_dirty = False
        return self._documents_cache
    
    @property
    def documents(self):
        """Lazy load documents."""
        return self._load_documents()
    
    def _save_documents(self):
        """Save all documents to the collection file."""
        if self._documents_cache is None:
            return
            
        with open(self.collection_file, 'w') as f:
            json.dump(self._documents_cache, f, separators=(',', ':'))  # Compact format to save space
        self._cache_dirty = False
    
    def _generate_id(self) -> str:
        """Generate a unique document ID."""
        return str(uuid.uuid4())
    
    def find(self, filter_dict: Optional[Dict] = None, projection: Optional[Dict] = None) -> Iterator[Dict]:
        """Find documents matching the filter."""
        filter_dict = filter_dict or {}
        
        for doc in self.documents.values():
            if self._matches_filter(doc, filter_dict):
                yield self._apply_projection(doc, projection)
    
    def find_one(self, filter_dict: Optional[Dict] = None, projection: Optional[Dict] = None) -> Optional[Dict]:
        """Find one document matching the filter."""
        for doc in self.find(filter_dict, projection):
            return doc
        return None
    
    def insert_one(self, document: Dict) -> 'InsertResult':
        """Insert a single document."""
        doc = document.copy()
        if "_id" not in doc:
            doc["_id"] = self._generate_id()
        
        self.documents[str(doc["_id"])] = doc
        self._cache_dirty = True
        self._save_documents()
        return InsertResult(doc["_id"])
    
    def insert_many(self, documents: List[Dict]) -> 'InsertManyResult':
        """Insert multiple documents efficiently."""
        inserted_ids = []
        for doc in documents:
            doc = doc.copy()
            if "_id" not in doc:
                doc["_id"] = self._generate_id()
            self.documents[str(doc["_id"])] = doc
            inserted_ids.append(doc["_id"])
        
        self._cache_dirty = True
        self._save_documents()
        return InsertManyResult(inserted_ids)
    
    def bulk_insert_streaming(self, documents: Iterator[Dict], batch_size: int = 10000) -> 'InsertManyResult':
        """
        Efficiently insert many documents in batches without loading everything into memory.
        This is optimized for the export process.
        """
        inserted_ids = []
        batch = []
        
        # Start with empty collection for bulk insert
        if not self.collection_file.exists():
            self._documents_cache = {}
            self._cache_dirty = True
        
        for doc in documents:
            doc = doc.copy()
            if "_id" not in doc:
                doc["_id"] = self._generate_id()
            
            batch.append(doc)
            inserted_ids.append(doc["_id"])
            
            # Write in batches to manage memory
            if len(batch) >= batch_size:
                for batch_doc in batch:
                    self.documents[str(batch_doc["_id"])] = batch_doc
                self._cache_dirty = True
                self._save_documents()
                batch = []
                print(f"      📈 Batch saved: {len(inserted_ids)} documents so far...")
        
        # Write remaining documents
        if batch:
            for batch_doc in batch:
                self.documents[str(batch_doc["_id"])] = batch_doc
            self._cache_dirty = True
            self._save_documents()
        
        return InsertManyResult(inserted_ids)
    
    def update_one(self, filter_dict: Dict, update: Dict) -> 'UpdateResult':
        """Update one document matching the filter."""
        for doc in self.documents.values():
            if self._matches_filter(doc, filter_dict):
                if self._apply_update(doc, update):
                    self._cache_dirty = True
                    self._save_documents()
                    return UpdateResult(1, 1)
                return UpdateResult(1, 0)
        return UpdateResult(0, 0)
    
    def update_many(self, filter_dict: Dict, update: Dict) -> 'UpdateResult':
        """Update all documents matching the filter."""
        matched = 0
        modified = 0
        for doc in self.documents.values():
            if self._matches_filter(doc, filter_dict):
                matched += 1
                if self._apply_update(doc, update):
                    modified += 1
        
        if modified > 0:
            self._cache_dirty = True
            self._save_documents()
        return UpdateResult(matched, modified)
    
    def delete_one(self, filter_dict: Dict) -> 'DeleteResult':
        """Delete one document matching the filter."""
        for doc_id, doc in list(self.documents.items()):
            if self._matches_filter(doc, filter_dict):
                del self.documents[doc_id]
                self._cache_dirty = True
                self._save_documents()
                return DeleteResult(1)
        return DeleteResult(0)
    
    def delete_many(self, filter_dict: Dict) -> 'DeleteResult':
        """Delete all documents matching the filter."""
        deleted_count = 0
        for doc_id, doc in list(self.documents.items()):
            if self._matches_filter(doc, filter_dict):
                del self.documents[doc_id]
                deleted_count += 1
        
        if deleted_count > 0:
            self._cache_dirty = True
            self._save_documents()
        return DeleteResult(deleted_count)
    
    def count_documents(self, filter_dict: Optional[Dict] = None) -> int:
        """Count documents matching the filter."""
        if filter_dict is None or filter_dict == {}:
            return len(self.documents)
        
        return sum(1 for _ in self.find(filter_dict))
    
    def distinct(self, field: str, filter_dict: Optional[Dict] = None) -> List[Any]:
        """Get distinct values for a field."""
        values = set()
        for doc in self.find(filter_dict):
            if field in doc:
                values.add(doc[field])
        return list(values)
    
    def _matches_filter(self, doc: Dict, filter_dict: Dict) -> bool:
        """Check if document matches the filter."""
        for key, value in filter_dict.items():
            if key.startswith("$"):
                # Handle MongoDB operators
                if key == "$or":
                    if not any(self._matches_filter(doc, condition) for condition in value):
                        return False
                elif key == "$and":
                    if not all(self._matches_filter(doc, condition) for condition in value):
                        return False
                # Add more operators as needed
            else:
                if isinstance(value, dict):
                    # Handle field operators
                    if "$in" in value:
                        if doc.get(key) not in value["$in"]:
                            return False
                    elif "$ne" in value:
                        if doc.get(key) == value["$ne"]:
                            return False
                    elif "$gt" in value:
                        if not (key in doc and doc[key] > value["$gt"]):
                            return False
                    elif "$gte" in value:
                        if not (key in doc and doc[key] >= value["$gte"]):
                            return False
                    elif "$lt" in value:
                        if not (key in doc and doc[key] < value["$lt"]):
                            return False
                    elif "$lte" in value:
                        if not (key in doc and doc[key] <= value["$lte"]):
                            return False
                    elif "$exists" in value:
                        exists = key in doc
                        if exists != value["$exists"]:
                            return False
                    # Add more field operators as needed
                else:
                    # Simple equality check
                    if doc.get(key) != value:
                        return False
        return True
    
    def _apply_update(self, doc: Dict, update: Dict) -> bool:
        """Apply update operators to document."""
        modified = False
        for operator, operations in update.items():
            if operator == "$set":
                for key, value in operations.items():
                    if doc.get(key) != value:
                        doc[key] = value
                        modified = True
            elif operator == "$unset":
                for key in operations:
                    if key in doc:
                        del doc[key]
                        modified = True
            elif operator == "$inc":
                for key, value in operations.items():
                    doc[key] = doc.get(key, 0) + value
                    modified = True
            elif operator == "$push":
                for key, value in operations.items():
                    if key not in doc:
                        doc[key] = []
                    doc[key].append(value)
                    modified = True
            elif operator == "$pull":
                for key, value in operations.items():
                    if key in doc and isinstance(doc[key], list):
                        original_length = len(doc[key])
                        doc[key] = [item for item in doc[key] if item != value]
                        if len(doc[key]) != original_length:
                            modified = True
            # Add more update operators as needed
        return modified
    
    def _apply_projection(self, doc: Dict, projection: Optional[Dict]) -> Dict:
        """Apply projection to document."""
        if not projection:
            return doc
        
        if projection.get("_id") == 1:
            return {"_id": doc["_id"]}
        
        result = {}
        for key, include in projection.items():
            if include == 1 and key in doc:
                result[key] = doc[key]
        return result if result else doc


class LocalDatabase:
    """Local file-based database that mimics MongoDB Database interface."""
    
    def __init__(self, storage: LocalStorage, db_name: str):
        self.storage = storage
        self.db_name = db_name
    
    def __getitem__(self, collection_name: str) -> LocalCollection:
        """Get a collection."""
        return LocalCollection(self.storage, self.db_name, collection_name)
    
    def __getattr__(self, collection_name: str) -> LocalCollection:
        """Get collection by name using attribute notation."""
        return LocalCollection(self.storage, self.db_name, collection_name)
    
    def drop_collection(self, collection_name: str) -> None:
        """Drop a collection."""
        collection_path = self.storage.get_collection_path(self.db_name, collection_name)
        if collection_path.exists():
            shutil.rmtree(collection_path)


class LocalClient:
    """Local file-based client that mimics MongoDB MongoClient interface."""
    
    def __init__(self, base_path: str = "data"):
        self.storage = LocalStorage(base_path)
    
    def __getitem__(self, db_name: str) -> LocalDatabase:
        """Get a database."""
        return LocalDatabase(self.storage, db_name)
    
    def __getattr__(self, db_name: str) -> LocalDatabase:
        """Get database by name using attribute notation."""
        return LocalDatabase(self.storage, db_name)
    
    def drop_database(self, db_name: str) -> None:
        """Drop a database."""
        db_path = self.storage.base_path / db_name
        if db_path.exists():
            shutil.rmtree(db_path)


# Result classes that mimic pymongo result objects
class InsertResult:
    def __init__(self, inserted_id: str):
        self.inserted_id = inserted_id


class InsertManyResult:
    def __init__(self, inserted_ids: List[str]):
        self.inserted_ids = inserted_ids


class UpdateResult:
    def __init__(self, matched_count: int, modified_count: int):
        self.matched_count = matched_count
        self.modified_count = modified_count


class DeleteResult:
    def __init__(self, deleted_count: int):
        self.deleted_count = deleted_count
