"""
Add Gemini Embeddings to Qdrant
==============================

This script reads Gemini embeddings from a JSON file and inserts them into a Qdrant collection.

Usage:
    python gemini_embedding.py

Requirements:
    - pip install qdrant-client
    - Qdrant running (local or cloud)
    - Environment variables set for Qdrant credentials
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Load environment variables
load_dotenv()

# Config - use environment variables for security
QDRANT_URL = os.getenv("QDRANT_URL", "localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Paths - fixed to use correct relative path
SCRIPT_DIR = Path(__file__).parent
EMBEDDINGS_PATH = (
    SCRIPT_DIR / "data" / "embeddings" / "gemini_embeddings_with_metadata.json"
)
COLLECTION_NAME = "gemini_vectors"
VECTOR_SIZE = 768  # Default for gemini-embedding-001 (updated from 1536)

# Validate input file exists
if not EMBEDDINGS_PATH.exists():
    raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_PATH}")

print(f"📁 Reading embeddings from: {EMBEDDINGS_PATH}")

# Load embeddings
try:
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        embeddings = data.get("embeddings", [])
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON in embeddings file: {e}")

if not embeddings:
    raise ValueError(f"No embeddings found in {EMBEDDINGS_PATH}")

print(f"📊 Found {len(embeddings)} embeddings to process")

# Validate vector size from first embedding
if embeddings:
    actual_vector_size = len(embeddings[0]["embedding"])
    if actual_vector_size != VECTOR_SIZE:
        print(
            f"⚠️  Warning: Expected vector size {VECTOR_SIZE}, but found {actual_vector_size}"
        )
        VECTOR_SIZE = actual_vector_size

# Connect to Qdrant
try:
    if QDRANT_URL.startswith("http"):
        # Cloud Qdrant
        if not QDRANT_API_KEY:
            raise ValueError(
                "QDRANT_API_KEY environment variable required for cloud Qdrant"
            )
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        # Local Qdrant
        client = QdrantClient(
            host=QDRANT_URL.split(":")[0], port=int(QDRANT_URL.split(":")[1])
        )

    print(f"🔗 Connected to Qdrant at: {QDRANT_URL}")
except Exception as e:
    raise ConnectionError(f"Failed to connect to Qdrant: {e}")

# Create collection if not exists
try:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"📦 Created/recreated collection: {COLLECTION_NAME}")
except Exception as e:
    raise RuntimeError(f"Failed to create collection: {e}")

# Prepare points
print("🔄 Preparing points for insertion...")
points = []
for idx, item in enumerate(embeddings):
    try:
        vector = item["embedding"]
        if len(vector) != VECTOR_SIZE:
            print(f"⚠️  Skipping item {item.get('id', idx)}: vector size mismatch")
            continue

        payload = {
            "id": item.get("id"),
            "content": item.get("content"),
            "content_type": item.get("content_type"),
            "source_url": item.get("source_url"),
            "metadata": item.get("metadata", {}),
        }
        points.append(PointStruct(id=idx, vector=vector, payload=payload))

        if (idx + 1) % 100 == 0:
            print(f"   Processed {idx + 1}/{len(embeddings)} items")

    except Exception as e:
        print(f"❌ Error processing item {idx}: {e}")
        continue

print(f"📋 Prepared {len(points)} points for insertion")

# Insert into Qdrant
try:
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(
        f"✅ Successfully inserted {len(points)} embeddings into Qdrant collection '{COLLECTION_NAME}'"
    )
except Exception as e:
    raise RuntimeError(f"Failed to insert embeddings: {e}")

# Verify insertion
try:
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"📊 Collection info: {collection_info.points_count} total points")
except Exception as e:
    print(f"⚠️  Could not verify collection info: {e}")
