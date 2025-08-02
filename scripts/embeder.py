import json
import os
import time
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable (more secure)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable not set. Please set it in your .env file"
    )

genai.configure(api_key=api_key)

# Paths - fixed to use correct relative path
SCRIPT_DIR = Path(__file__).parent
INPUT_PATH = SCRIPT_DIR / "data" / "rag_ready" / "rag_ready_data.json"
OUTPUT_PATH = (
    SCRIPT_DIR / "data" / "embeddings" / "gemini_embeddings_with_metadata.json"
)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Validate input file exists
if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

print(f"📁 Reading from: {INPUT_PATH}")
print(f"📁 Writing to: {OUTPUT_PATH}")

# Load RAG-ready content
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"📊 Found {len(data)} items to process")

embeddings_output = []
errors = []

for i, item in enumerate(data):
    text = item["content"]

    # Skip empty content
    if not text or not text.strip():
        print(f"⚠️  Skipping item {item['id']}: empty content")
        continue

    try:
        print(f"🔄 Processing item {i+1}/{len(data)}: {item['id']}")

        # Call Gemini's embedding endpoint
        response = genai.embed_content(
            model="models/embedding-001", content=text, task_type="retrieval_document"
        )
        embedding = response["embedding"]

        embeddings_output.append(
            {
                "id": item["id"],
                "embedding": embedding,
                "content": text,
                "content_type": "text",
                "source_url": item["metadata"].get("original_url", ""),
                "metadata": item["metadata"],
            }
        )

        # Add small delay to avoid rate limiting
        time.sleep(0.1)

    except Exception as e:
        error_msg = f"Error embedding item {item['id']}: {e}"
        print(f"❌ {error_msg}")
        errors.append(error_msg)

# Save to file
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump({"embeddings": embeddings_output}, f, indent=2)

print(f"✅ Saved {len(embeddings_output)} embeddings to {OUTPUT_PATH}")
if errors:
    print(f"⚠️  {len(errors)} errors occurred during processing")
    for error in errors[:5]:  # Show first 5 errors
        print(f"   - {error}")
