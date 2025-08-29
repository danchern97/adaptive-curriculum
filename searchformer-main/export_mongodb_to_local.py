#!/usr/bin/env python3
"""
Export data from MongoDB to local storage format.

This script connects to a running MongoDB instance and exports all data
to the local storage format used by searchformer.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add searchformer to path
sys.path.insert(0, 'searchformer')

def export_mongodb_to_local_storage(output_dir: str, mongodb_uri: str = "mongodb://localhost:27017"):
    """Export all data from MongoDB to local storage format."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Exporting MongoDB data to local storage: {output_dir}")
    
    # First, temporarily enable MongoDB to connect and read the data
    os.environ['USE_MONGODB'] = 'true'
    os.environ['MONGODB_URI'] = mongodb_uri
    
    try:
        from searchformer.utils import mongodb_client
        from pymongo import MongoClient
        
        # Connect to MongoDB
        mongo_client = mongodb_client()
        if not isinstance(mongo_client, MongoClient):
            print("❌ Failed to connect to MongoDB")
            return False
        
        print("✅ Connected to MongoDB")
        
        # List all databases
        db_names = mongo_client.list_database_names()
        searchformer_dbs = [db for db in db_names if db in ['tokenSeqDB', 'trainDB', 'ckptDB', 'rolloutDB']]
        
        if not searchformer_dbs:
            print("⚠️ No searchformer databases found. Available databases:")
            for db_name in db_names:
                print(f"  - {db_name}")
            return False
        
        print(f"📁 Found searchformer databases: {searchformer_dbs}")
        
        # Now switch to local storage for writing
        os.environ['USE_MONGODB'] = 'false'
        os.environ['LOCAL_DATA_PATH'] = str(output_dir)
        
        # Import local storage after setting environment
        from searchformer.local_storage import LocalClient
        
        local_client = LocalClient(str(output_dir))
        print("✅ Local storage client created")
        
        total_exported = 0
        
        # Export each database
        for db_name in searchformer_dbs:
            print(f"\n📂 Processing database: {db_name}")
            
            mongo_db = mongo_client[db_name]
            local_db = local_client[db_name]
            
            # Get all collections in this database
            collection_names = mongo_db.list_collection_names()
            
            if not collection_names:
                print(f"  ⚠️ No collections found in {db_name}")
                continue
            
            print(f"  📄 Found collections: {collection_names}")
            
            # Export each collection
            for collection_name in collection_names:
                print(f"    🔄 Exporting {collection_name}...")
                
                mongo_collection = mongo_db[collection_name]
                local_collection = local_db[collection_name]
                
                # Get document count
                doc_count = mongo_collection.count_documents({})
                print(f"      📊 {doc_count} documents to export")
                
                if doc_count == 0:
                    continue
                
                # Export documents efficiently using streaming bulk insert
                def document_iterator():
                    for doc in mongo_collection.find():
                        yield doc
                
                try:
                    result = local_collection.bulk_insert_streaming(document_iterator(), batch_size=10000)
                    exported_count = len(result.inserted_ids)
                    total_exported += exported_count
                    print(f"      ✅ Exported {exported_count} documents from {collection_name}")
                
                except Exception as e:
                    print(f"      ❌ Failed to export collection {collection_name}: {e}")
                    # Fallback to individual inserts if bulk fails
                    print(f"      🔄 Falling back to individual inserts...")
                    exported_count = 0
                    for doc in mongo_collection.find():
                        try:
                            local_collection.insert_one(doc)
                            exported_count += 1
                            total_exported += 1
                            
                            if exported_count % 1000 == 0:
                                print(f"      📈 Exported {exported_count}/{doc_count} documents...")
                        
                        except Exception as e:
                            print(f"      ⚠️ Failed to export document {doc.get('_id', 'unknown')}: {e}")
                    
                    print(f"      ✅ Exported {exported_count} documents from {collection_name} (fallback mode)")
        
        print(f"\n🎉 Export complete!")
        print(f"📊 Total documents exported: {total_exported}")
        print(f"📁 Data saved to: {output_dir.absolute()}")
        
        return True
        
    except ImportError as e:
        print(f"❌ MongoDB not available: {e}")
        print("Make sure pymongo is installed: pip install pymongo")
        return False
    
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export MongoDB data to local storage")
    parser.add_argument("--output-dir", default="./exported_data", 
                       help="Directory for local storage data (default: ./exported_data)")
    parser.add_argument("--mongodb-uri", default="mongodb://localhost:27017",
                       help="MongoDB connection URI (default: mongodb://localhost:27017)")
    
    args = parser.parse_args()
    
    print("🔄 MongoDB to Local Storage Exporter")
    print("=" * 40)
    print(f"📍 MongoDB URI: {args.mongodb_uri}")
    print(f"📁 Output directory: {args.output_dir}")
    print()
    
    success = export_mongodb_to_local_storage(args.output_dir, args.mongodb_uri)
    
    if success:
        print(f"\n✨ Success! To use this data with searchformer:")
        print(f"   export LOCAL_DATA_PATH={Path(args.output_dir).absolute()}")
        print(f"   # Or in PowerShell: $env:LOCAL_DATA_PATH = '{Path(args.output_dir).absolute()}'")
        return 0
    else:
        print(f"\n❌ Export failed. Make sure:")
        print(f"   1. MongoDB is running on {args.mongodb_uri}")
        print(f"   2. Data has been imported with mongorestore")
        print(f"   3. pymongo is installed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
