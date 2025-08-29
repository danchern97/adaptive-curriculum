#!/usr/bin/env python3
"""
Convert MongoDB archive files to local storage format.

This script extracts data from MongoDB archive files (like searchformer.gz) 
and converts it to the local storage format used by searchformer.
"""

import os
import sys
import gzip
import json
import subprocess
import tempfile
import shutil
from pathlib import Path

# Add searchformer to path
sys.path.insert(0, 'searchformer')

def extract_mongodb_archive(archive_path: str, output_dir: str) -> None:
    """Extract MongoDB archive to JSON files using mongorestore and mongoexport."""
    
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive file not found: {archive_path}")
    
    print(f"📁 Extracting {archive_path.name}...")
    
    # Create temporary directory for MongoDB restore
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_db_dir = Path(temp_dir) / "mongod_data"
        temp_db_dir.mkdir()
        
        # Start temporary MongoDB instance
        print("🔧 Starting temporary MongoDB instance...")
        mongod_port = "27018"  # Use different port to avoid conflicts
        
        mongod_process = subprocess.Popen([
            "mongod",
            "--dbpath", str(temp_db_dir),
            "--port", mongod_port,
            "--nojournal",
            "--smallfiles",
            "--quiet"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        try:
            # Wait a moment for MongoDB to start
            import time
            time.sleep(3)
            
            # Restore the archive
            print("📦 Restoring archive to temporary MongoDB...")
            restore_cmd = [
                "mongorestore",
                "--host", f"localhost:{mongod_port}",
                "--gzip",
                f"--archive={archive_path}"
            ]
            
            result = subprocess.run(restore_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ mongorestore failed: {result.stderr}")
                return
            
            # List databases to see what was imported
            list_dbs_cmd = [
                "mongo",
                "--host", f"localhost:{mongod_port}",
                "--eval", "printjson(db.adminCommand('listDatabases'))"
            ]
            
            result = subprocess.run(list_dbs_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("📋 Imported databases:")
                print(result.stdout)
            
            # Export each collection to JSON
            # Common database names in searchformer archives
            db_names = ["tokenSeqDB", "trainDB", "ckptDB"]
            
            for db_name in db_names:
                print(f"🔍 Checking database: {db_name}")
                
                # List collections in database
                list_collections_cmd = [
                    "mongo",
                    "--host", f"localhost:{mongod_port}",
                    db_name,
                    "--eval", "printjson(db.getCollectionNames())"
                ]
                
                result = subprocess.run(list_collections_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    continue
                
                # Parse collection names from output
                collections_output = result.stdout
                if "[ ]" in collections_output or not collections_output.strip():
                    continue
                
                # Extract collections and export each one
                try:
                    # Simple parsing - look for array-like structure
                    import re
                    collections_match = re.search(r'\[(.*?)\]', collections_output, re.DOTALL)
                    if collections_match:
                        collections_str = collections_match.group(1)
                        collections = [c.strip().strip('"') for c in collections_str.split(',') if c.strip()]
                        
                        for collection in collections:
                            if not collection:
                                continue
                                
                            print(f"💾 Exporting {db_name}.{collection}...")
                            
                            # Create output directory structure
                            collection_dir = output_dir / db_name / collection
                            collection_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Export collection to JSON
                            export_cmd = [
                                "mongoexport",
                                "--host", f"localhost:{mongod_port}",
                                "--db", db_name,
                                "--collection", collection,
                                "--out", str(collection_dir / f"{collection}.json")
                            ]
                            
                            export_result = subprocess.run(export_cmd, capture_output=True, text=True)
                            if export_result.returncode == 0:
                                print(f"✅ Exported {collection} ({export_result.stdout.strip()})")
                            else:
                                print(f"❌ Failed to export {collection}: {export_result.stderr}")
                
                except Exception as e:
                    print(f"⚠️ Error parsing collections for {db_name}: {e}")
        
        finally:
            # Stop MongoDB
            print("🔌 Stopping temporary MongoDB...")
            mongod_process.terminate()
            mongod_process.wait()


def convert_json_to_local_storage(json_dir: str, local_storage_dir: str) -> None:
    """Convert exported JSON files to local storage format."""
    
    json_dir = Path(json_dir)
    local_storage_dir = Path(local_storage_dir)
    
    print(f"🔄 Converting JSON exports to local storage format...")
    
    # Set up local storage
    os.environ['LOCAL_DATA_PATH'] = str(local_storage_dir)
    
    from searchformer.utils import mongodb_client
    from searchformer.local_storage import LocalClient
    
    client = mongodb_client()
    if not isinstance(client, LocalClient):
        print("❌ Expected LocalClient, but got different client type")
        return
    
    # Process each database directory
    for db_dir in json_dir.iterdir():
        if not db_dir.is_dir():
            continue
            
        db_name = db_dir.name
        print(f"📂 Processing database: {db_name}")
        
        db = client[db_name]
        
        # Process each collection directory
        for collection_dir in db_dir.iterdir():
            if not collection_dir.is_dir():
                continue
                
            collection_name = collection_dir.name
            print(f"  📄 Processing collection: {collection_name}")
            
            collection = db[collection_name]
            
            # Find JSON export file
            json_file = collection_dir / f"{collection_name}.json"
            if not json_file.exists():
                print(f"    ⚠️ No JSON file found for {collection_name}")
                continue
            
            # Read and import documents
            imported_count = 0
            try:
                with open(json_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            doc = json.loads(line)
                            collection.insert_one(doc)
                            imported_count += 1
                            
                            if imported_count % 1000 == 0:
                                print(f"    📊 Imported {imported_count} documents...")
                        
                        except json.JSONDecodeError as e:
                            print(f"    ⚠️ Skipping invalid JSON line: {e}")
                            continue
                
                print(f"    ✅ Imported {imported_count} documents to {collection_name}")
                
            except Exception as e:
                print(f"    ❌ Error importing {collection_name}: {e}")


def main():
    """Main conversion function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert MongoDB archives to local storage")
    parser.add_argument("archive", help="Path to MongoDB archive file (e.g., searchformer.gz)")
    parser.add_argument("--output-dir", default="./converted_data", 
                       help="Directory for local storage data (default: ./converted_data)")
    parser.add_argument("--temp-json", default="./temp_json",
                       help="Temporary directory for JSON exports (default: ./temp_json)")
    
    args = parser.parse_args()
    
    archive_path = Path(args.archive)
    output_dir = Path(args.output_dir)
    temp_json_dir = Path(args.temp_json)
    
    if not archive_path.exists():
        print(f"❌ Archive file not found: {archive_path}")
        return 1
    
    try:
        # Step 1: Extract MongoDB archive to JSON files
        extract_mongodb_archive(str(archive_path), str(temp_json_dir))
        
        # Step 2: Convert JSON files to local storage
        convert_json_to_local_storage(str(temp_json_dir), str(output_dir))
        
        print(f"\n🎉 Conversion complete!")
        print(f"📁 Local storage data available in: {output_dir.absolute()}")
        print(f"\nTo use this data, set environment variable:")
        print(f"  LOCAL_DATA_PATH={output_dir.absolute()}")
        
        # Clean up temporary JSON files
        if temp_json_dir.exists():
            shutil.rmtree(temp_json_dir)
            print(f"🧹 Cleaned up temporary files")
        
        return 0
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
