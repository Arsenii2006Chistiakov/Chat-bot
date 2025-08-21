#!/usr/bin/env python3
"""
Script to update TOP_TIKTOK_SOUNDS collection with trend_countries from TOP_TIKTOK_TRENDS.

For every document in TOP_TIKTOK_TRENDS, this script:
1. Extracts song_id and countries list
2. Finds the corresponding document in TOP_TIKTOK_SOUNDS by song_id
3. Updates it with trend_countries field containing the same countries list
"""

import os
import sys
from typing import List, Dict, Any
import pymongo
from pymongo import MongoClient
from datetime import datetime

def get_mongo_connection():
    """Get MongoDB connection from environment or use default."""
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    try:
        client = MongoClient(mongo_uri)
        # Test connection
        client.admin.command('ping')
        print(f"✅ Connected to MongoDB at {mongo_uri}")
        return client
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        sys.exit(1)

def get_collections(client):
    """Get the required collections."""
    try:
        db = client.HUGO_MODEL_DB
        trends_collection = db.TOP_TIKTOK_TRENDS
        sounds_collection = db.TOP_TIKTOK_SOUNDS
        
        # Test collections exist
        trends_count = trends_collection.count_documents({})
        sounds_count = sounds_collection.count_documents({})
        
        print(f"📊 TOP_TIKTOK_TRENDS: {trends_count} documents")
        print(f"📊 TOP_TIKTOK_SOUNDS: {sounds_count} documents")
        
        return trends_collection, sounds_collection
    except Exception as e:
        print(f"❌ Failed to access collections: {e}")
        sys.exit(1)

def extract_trend_data(trends_collection):
    """Extract song_id and countries from TOP_TIKTOK_TRENDS."""
    trend_data = []
    
    try:
        # Find all documents with countries field
        cursor = trends_collection.find({"countries": {"$exists": True, "$ne": []}})
        
        for doc in cursor:
            song_id = doc.get("song_id")
            countries = doc.get("countries", [])
            
            if song_id and countries:
                trend_data.append({
                    "song_id": song_id,
                    "countries": countries,
                    "_id": doc.get("_id")
                })
        
        print(f"🔍 Found {len(trend_data)} trends with countries data")
        return trend_data
        
    except Exception as e:
        print(f"❌ Error extracting trend data: {e}")
        return []

def update_sounds_with_trend_countries(sounds_collection, trend_data):
    """Update TOP_TIKTOK_SOUNDS documents with trend_countries."""
    updated_count = 0
    not_found_count = 0
    errors = []
    
    print(f"\n🔄 Starting update process for {len(trend_data)} trends...")
    
    for trend in trend_data:
        song_id = trend["song_id"]
        countries = trend["countries"]
        
        try:
            # Find the corresponding document in TOP_TIKTOK_SOUNDS
            result = sounds_collection.update_one(
                {"song_id": song_id},
                {
                    "$set": {
                        "trend_countries": countries,
                        "trend_countries_updated_at": datetime.utcnow()
                    }
                }
            )
            
            if result.matched_count > 0:
                if result.modified_count > 0:
                    updated_count += 1
                    print(f"✅ Updated song_id: {song_id} with {len(countries)} countries")
                else:
                    print(f"ℹ️  Song_id: {song_id} already had trend_countries (no change)")
            else:
                not_found_count += 1
                print(f"⚠️  Song_id: {song_id} not found in TOP_TIKTOK_SOUNDS")
                
        except Exception as e:
            error_msg = f"❌ Error updating song_id {song_id}: {e}"
            print(error_msg)
            errors.append({"song_id": song_id, "error": str(e)})
    
    return updated_count, not_found_count, errors

def main():
    """Main execution function."""
    print("🚀 Starting trend countries update script...")
    print("=" * 50)
    
    # Connect to MongoDB
    client = get_mongo_connection()
    
    try:
        # Get collections
        trends_collection, sounds_collection = get_collections(client)
        
        # Extract trend data
        trend_data = extract_trend_data(trends_collection)
        
        if not trend_data:
            print("❌ No trend data found. Exiting.")
            return
        
        # Update sounds collection
        updated_count, not_found_count, errors = update_sounds_with_trend_countries(
            sounds_collection, trend_data
        )
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 UPDATE SUMMARY")
        print("=" * 50)
        print(f"✅ Successfully updated: {updated_count}")
        print(f"⚠️  Songs not found: {not_found_count}")
        print(f"❌ Errors: {len(errors)}")
        
        if errors:
            print(f"\n❌ Error details:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  - {error['song_id']}: {error['error']}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
        
        print(f"\n🎯 Total trends processed: {len(trend_data)}")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        client.close()
        print("\n🔌 MongoDB connection closed")

if __name__ == "__main__":
    main()
