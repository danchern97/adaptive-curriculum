#!/usr/bin/env python3
"""
Example script showing how to use searchformer with local storage instead of MongoDB.
"""

import os
import sys

# Local storage is now the default - no environment variables needed!
# Optionally set a custom data directory
os.environ['LOCAL_DATA_PATH'] = './local_data'

# Ensure the searchformer module is in the path
sys.path.insert(0, 'searchformer')

if __name__ == "__main__":
    print("🗂️  Searchformer Local Storage Example")
    print("=" * 40)
    
    # Test basic client and database operations
    print("\n1. Creating client and database...")
    from searchformer.utils import mongodb_client
    
    client = mongodb_client()
    print(f"✓ Client created: {type(client)}")
    
    db = client.example_database
    collection = db.example_collection
    print("✓ Database and collection created")
    
    # Test document operations
    print("\n2. Testing document operations...")
    
    # Insert a document
    doc = {
        "name": "Example Document",
        "type": "test",
        "value": 42,
        "tags": ["example", "demo"]
    }
    result = collection.insert_one(doc)
    print(f"✓ Document inserted with ID: {result.inserted_id}")
    
    # Find the document
    found_doc = collection.find_one({"name": "Example Document"})
    print(f"✓ Document retrieved: {found_doc['name']}")
    
    # Update the document
    collection.update_one(
        {"name": "Example Document"},
        {"$set": {"value": 100, "updated": True}}
    )
    print("✓ Document updated")
    
    # Find with filters
    docs_with_tags = list(collection.find({"tags": {"$in": ["demo"]}}))
    print(f"✓ Found {len(docs_with_tags)} documents with 'demo' tag")
    
    # Test with TokenizedDataset (if vocabulary exists)
    print("\n3. Testing integration with searchformer modules...")
    try:
        from searchformer.trace import TokenizedDataset
        
        # This might fail if no vocabulary exists, which is expected
        print("✓ TokenizedDataset class imported successfully")
        print("  (Note: Creating an actual dataset requires existing vocabulary)")
        
    except Exception as e:
        print(f"ℹ️  TokenizedDataset test skipped: {e}")
    
    # Show data location
    print(f"\n📁 Data stored in: {os.path.abspath('./local_data')}")
    print("\n🎉 Example completed successfully!")
    print("\nSearchformer now uses local storage by default!")
    print("To use MongoDB instead:")
    print("  1. Set environment variable: USE_MONGODB=true")
    print("  2. Optionally set MongoDB URI: MONGODB_URI=mongodb://localhost:27017")
    print("  3. Make sure pymongo is installed")
    print("\nFor local storage (default):")
    print("  - No setup required!")
    print("  - Optionally set data path: LOCAL_DATA_PATH=/path/to/data")
