#!/usr/bin/env python3
"""
Test script to verify local storage implementation works correctly.
"""

import os
import sys
import tempfile
import shutil

# Local storage is now the default, no need to set USE_LOCAL_STORAGE
# Just set the data path for testing

def test_local_storage():
    """Test basic local storage functionality."""
    print("Testing local storage implementation...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ['LOCAL_DATA_PATH'] = temp_dir
        
        try:
            from searchformer.utils import mongodb_client
            from searchformer.local_storage import LocalStorage
            
            # Test client creation
            client = mongodb_client()
            print(f"✓ Client created: {type(client)}")
            
            # Verify it's using local storage
            assert hasattr(client, 'storage'), "Client should have storage attribute for local storage"
            assert isinstance(client.storage, LocalStorage), "Storage should be LocalStorage instance"
            print("✓ Using local storage backend")
            
            # Test database and collection creation
            db = client.test_database
            collection = db.test_collection
            print("✓ Database and collection created")
            
            # Test basic operations
            # Insert a document
            doc = {"name": "test", "value": 42}
            result = collection.insert_one(doc)
            print(f"✓ Document inserted with ID: {result.inserted_id}")
            
            # Find the document
            found_doc = collection.find_one({"name": "test"})
            assert found_doc is not None, "Document should be found"
            assert found_doc["value"] == 42, "Document value should match"
            print("✓ Document found and verified")
            
            # Update the document
            collection.update_one({"name": "test"}, {"$set": {"value": 100}})
            updated_doc = collection.find_one({"name": "test"})
            assert updated_doc["value"] == 100, "Document should be updated"
            print("✓ Document updated successfully")
            
            # Test find with multiple documents
            collection.insert_one({"name": "test2", "value": 200})
            docs = list(collection.find({}))
            assert len(docs) == 2, "Should find 2 documents"
            print("✓ Multiple documents handled correctly")
            
            # Test GridFS functionality
            try:
                from searchformer.local_gridfs import LocalGridFS
                
                gridfs = LocalGridFS(client.storage, "test_gridfs")
                test_data = b"This is test binary data for GridFS replacement"
                
                # Store binary data
                file_id = gridfs.put(test_data, "test_file_id", filename="test_file.bin")
                print(f"✓ GridFS file stored with ID: {file_id}")
                
                # Retrieve binary data
                grid_file = gridfs.get(file_id)
                retrieved_data = grid_file.read()
                assert retrieved_data == test_data, "Retrieved data should match original"
                print("✓ GridFS file retrieved and verified")
                
                # Check file exists
                assert gridfs.exists(file_id), "File should exist"
                print("✓ GridFS file existence check works")
                
            except ImportError as e:
                print(f"⚠ GridFS test skipped: {e}")
            
            print("\n🎉 All local storage tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_imports():
    """Test that all searchformer modules can be imported with local storage."""
    print("\nTesting module imports...")
    
    try:
        from searchformer import utils
        print("✓ utils imported")
        
        from searchformer import trace
        print("✓ trace imported")
        
        from searchformer import train
        print("✓ train imported")
        
        from searchformer import rollout
        print("✓ rollout imported")
        
        print("✓ All modules imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting local storage tests...\n")
    
    success = True
    success &= test_imports()
    success &= test_local_storage()
    
    if success:
        print("\n🎉 All tests passed! Local storage implementation is working correctly.")
        print("\nSearchformer now uses local storage by default!")
        print("To use MongoDB instead, set these environment variables:")
        print("  export USE_MONGODB=true")
        print("  export MONGODB_URI=mongodb://localhost:27017")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
