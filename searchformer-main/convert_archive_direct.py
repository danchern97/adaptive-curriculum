#!/usr/bin/env python3
"""
Direct MongoDB archive to local storage converter.

This script directly parses MongoDB archive files and converts them to local storage
without requiring a running MongoDB instance.
"""

import os
import sys
import gzip
import json
import struct
from pathlib import Path
from typing import Dict, Any, Iterator

# Add searchformer to path
sys.path.insert(0, 'searchformer')


def read_bson_from_archive(archive_path: str) -> Iterator[Dict[str, Any]]:
    """
    Read BSON documents directly from MongoDB archive.
    
    Note: This is a simplified BSON parser and may not handle all BSON types.
    For complex archives, use the mongorestore method instead.
    """
    
    def read_cstring(data: bytes, offset: int) -> tuple[str, int]:
        """Read null-terminated string from BSON data."""
        end = data.find(b'\x00', offset)
        if end == -1:
            raise ValueError("Invalid BSON: unterminated string")
        return data[offset:end].decode('utf-8'), end + 1
    
    def read_int32(data: bytes, offset: int) -> tuple[int, int]:
        """Read 32-bit integer from BSON data."""
        return struct.unpack('<i', data[offset:offset+4])[0], offset + 4
    
    def read_int64(data: bytes, offset: int) -> tuple[int, int]:
        """Read 64-bit integer from BSON data."""
        return struct.unpack('<q', data[offset:offset+8])[0], offset + 8
    
    def read_double(data: bytes, offset: int) -> tuple[float, int]:
        """Read double from BSON data."""
        return struct.unpack('<d', data[offset:offset+8])[0], offset + 8
    
    def parse_bson_document(data: bytes, offset: int = 0) -> Dict[str, Any]:
        """Parse a single BSON document."""
        doc = {}
        doc_size, offset = read_int32(data, offset)
        end_offset = offset + doc_size - 4
        
        while offset < end_offset - 1:
            element_type = data[offset]
            offset += 1
            
            if element_type == 0:  # End of document
                break
            
            field_name, offset = read_cstring(data, offset)
            
            if element_type == 0x01:  # Double
                value, offset = read_double(data, offset)
            elif element_type == 0x02:  # String
                str_len, offset = read_int32(data, offset)
                value = data[offset:offset+str_len-1].decode('utf-8')
                offset += str_len
            elif element_type == 0x03:  # Document
                value = parse_bson_document(data, offset)
                doc_size, _ = read_int32(data, offset)
                offset += doc_size
            elif element_type == 0x04:  # Array
                value = parse_bson_document(data, offset)
                doc_size, _ = read_int32(data, offset)
                offset += doc_size
                # Convert document to list
                value = [value[str(i)] for i in range(len(value))]
            elif element_type == 0x08:  # Boolean
                value = data[offset] != 0
                offset += 1
            elif element_type == 0x10:  # 32-bit integer
                value, offset = read_int32(data, offset)
            elif element_type == 0x12:  # 64-bit integer
                value, offset = read_int64(data, offset)
            else:
                # Skip unknown types
                print(f"Warning: Skipping unknown BSON type {element_type:02x} for field {field_name}")
                break
            
            doc[field_name] = value
        
        return doc
    
    # Read archive file
    with gzip.open(archive_path, 'rb') as f:
        data = f.read()
    
    # Parse BSON documents
    offset = 0
    while offset < len(data):
        try:
            if offset + 4 >= len(data):
                break
            
            doc_size, _ = read_int32(data, offset)
            if doc_size <= 0 or offset + doc_size > len(data):
                break
            
            doc = parse_bson_document(data, offset)
            yield doc
            
            offset += doc_size
            
        except Exception as e:
            print(f"Warning: Error parsing document at offset {offset}: {e}")
            break


def convert_archive_direct(archive_path: str, output_dir: str) -> None:
    """Convert MongoDB archive directly to local storage."""
    
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive file not found: {archive_path}")
    
    print(f"📦 Converting {archive_path.name} directly to local storage...")
    
    # Set up local storage
    os.environ['LOCAL_DATA_PATH'] = str(output_dir)
    
    from searchformer.utils import mongodb_client
    from searchformer.local_storage import LocalClient
    
    client = mongodb_client()
    if not isinstance(client, LocalClient):
        print("❌ Expected LocalClient")
        return
    
    # Parse documents from archive
    documents = list(read_bson_from_archive(str(archive_path)))
    
    if not documents:
        print("⚠️ No documents found in archive. This might be a complex BSON format.")
        print("Try using the mongorestore method instead.")
        return
    
    print(f"📄 Found {len(documents)} documents")
    
    # Group documents by database and collection
    # This is a heuristic based on common searchformer patterns
    grouped_docs = {}
    
    for doc in documents:
        # Try to infer database and collection from document structure
        db_name = "tokenSeqDB"  # Default for searchformer data
        collection_name = "unknown"
        
        # Look for clues in the document
        if "_id" in doc:
            if isinstance(doc["_id"], str) and "." in doc["_id"]:
                # Pattern like "dataset.name" suggests collection name
                collection_name = doc["_id"].split(".")[0]
            elif "vocabulary" in str(doc.get("_id", "")):
                collection_name = "vocabulary"
        
        # Check for other identifying fields
        if "vocabulary" in doc:
            collection_name = "vocabulary"
        elif "seq" in doc or "tokens" in doc:
            collection_name = "sequences"
        elif "checkpoint" in doc or "model" in doc:
            db_name = "ckptDB"
            collection_name = "checkpoints"
        
        # Group documents
        if db_name not in grouped_docs:
            grouped_docs[db_name] = {}
        if collection_name not in grouped_docs[db_name]:
            grouped_docs[db_name][collection_name] = []
        
        grouped_docs[db_name][collection_name].append(doc)
    
    # Import to local storage
    total_imported = 0
    for db_name, collections in grouped_docs.items():
        print(f"📂 Importing to database: {db_name}")
        db = client[db_name]
        
        for collection_name, docs in collections.items():
            print(f"  📄 Importing {len(docs)} documents to collection: {collection_name}")
            collection = db[collection_name]
            
            for doc in docs:
                try:
                    collection.insert_one(doc)
                    total_imported += 1
                except Exception as e:
                    print(f"    ⚠️ Failed to import document: {e}")
    
    print(f"✅ Successfully imported {total_imported} documents")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert MongoDB archive to local storage (direct method)")
    parser.add_argument("archive", help="Path to MongoDB archive file (e.g., searchformer.gz)")
    parser.add_argument("--output-dir", default="./local_data", 
                       help="Directory for local storage data (default: ./local_data)")
    
    args = parser.parse_args()
    
    try:
        convert_archive_direct(args.archive, args.output_dir)
        
        print(f"\n🎉 Conversion complete!")
        print(f"📁 Data available in: {Path(args.output_dir).absolute()}")
        print(f"\nTo use this data, set environment variable:")
        print(f"  LOCAL_DATA_PATH={Path(args.output_dir).absolute()}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        print("\n💡 If this direct method fails, try the full MongoDB conversion:")
        print(f"  python convert_mongodb_archive.py {args.archive}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
