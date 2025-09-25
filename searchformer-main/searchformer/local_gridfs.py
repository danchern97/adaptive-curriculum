# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import io
import pickle
from pathlib import Path
from typing import Any, Dict, Optional


class LocalGridFS:
    """Local file-based replacement for GridFS."""
    
    def __init__(self, storage, collection: str):
        """Initialize local GridFS.
        
        Args:
            storage: LocalStorage instance
            collection: Collection name (used as subdirectory)
        """
        self.storage = storage
        self.collection = collection
        self.base_path = storage.base_path / "ckptDB" / collection
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def put(self, data: bytes, _id: str, **kwargs) -> str:
        """Store binary data with given ID.
        
        Args:
            data: Binary data to store
            _id: Unique identifier
            
        Returns:
            str: The ID of the stored file
        """
        file_path = self.base_path / f"{_id}.pkl"
        with open(file_path, 'wb') as f:
            f.write(data)
        return _id
    
    def get(self, file_id: str) -> 'LocalGridFile':
        """Retrieve data by ID.
        
        Args:
            file_id: ID of the file to retrieve
            
        Returns:
            LocalGridFile: File-like object containing the data
        """
        file_path = self.base_path / f"{file_id}.pkl"
        if not file_path.exists():
            raise FileNotFoundError(f"No file found with id {file_id}")
        
        with open(file_path, 'rb') as f:
            data = f.read()
        
        return LocalGridFile(data)
    
    def delete(self, file_id: str) -> None:
        """Delete file by ID.
        
        Args:
            file_id: ID of the file to delete
        """
        file_path = self.base_path / f"{file_id}.pkl"
        if file_path.exists():
            file_path.unlink()
    
    def exists(self, document_or_id: str) -> bool:
        """Check if file exists.
        
        Args:
            document_or_id: ID of the file to check
            
        Returns:
            bool: True if file exists, False otherwise
        """
        file_path = self.base_path / f"{document_or_id}.pkl"
        return file_path.exists()


class LocalGridFile:
    """Local file-like object that mimics GridFS GridOut."""
    
    def __init__(self, data: bytes):
        """Initialize with binary data.
        
        Args:
            data: Binary data
        """
        self._data = data
    
    def read(self) -> bytes:
        """Read the binary data.
        
        Returns:
            bytes: The stored binary data
        """
        return self._data


def create_local_gridfs(storage, collection: str) -> LocalGridFS:
    """Create a local GridFS instance.
    
    Args:
        storage: LocalStorage instance
        collection: Collection name
        
    Returns:
        LocalGridFS: Local GridFS instance
    """
    return LocalGridFS(storage, collection)
