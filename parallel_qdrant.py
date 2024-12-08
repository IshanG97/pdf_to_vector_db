from concurrent.futures import ThreadPoolExecutor
from qdrant_client import QdrantClient
import json
from time import sleep, time
from datetime import datetime

# Connect to Qdrant (with extended timeout)
client = QdrantClient(url="http://localhost:6333", timeout=600)

# Load embeddings from a JSON file
with open("image_embeddings.json", "r") as f:
    embeddings = json.load(f)

# Define collection name and vector size
collection_name = "image_embeddings"
vector_size = len(embeddings[0]["embedding"])  # Dimensionality of vectors
print(f"Vector size used for collection creation: {vector_size}")

if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config={"size": vector_size, "distance": "Cosine"}
)

# Prepare vectors and payloads for upload
points = [
    {
        "id": idx,  # Unique ID for each vector
        "vector": item["embedding"],  # Embedding vector
        "payload": {  # Optional metadata
            "path": item["path"],
            **item.get("metadata", {})
        }
    }
    for idx, item in enumerate(embeddings)
]

# Logging function to print timestamped messages
def log_progress(message):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def upload_batch(start, end):
    batch = points[start:end]
    log_progress(f"Starting upload for batch {start // batch_size + 1} (points {start}-{end})")
    
    print("Batch validation:")
    for idx, point in enumerate(batch[:5]):  # Check first 5 points
        print(f"  Point {idx + 1} ID: {point['id']}")
        print(f"  Vector size: {len(point['vector'])}")
        print(f"  Vector sample: {point['vector'][:5]}")  # Print first 5 elements
        print(f"  Payload: {point['payload']}")

    retries = 3
    for attempt in range(retries):
        try:
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
            log_progress(f"Successfully uploaded batch {start // batch_size + 1}")
            return
        except Exception as e:
            log_progress(f"Error in batch {start // batch_size + 1}, attempt {attempt + 1}: {e}")
            sleep(2)
    log_progress(f"Failed to upload batch {start // batch_size + 1} after {retries} retries.")


# Batch size
batch_size = 2000  # Adjust as needed

# Progress tracking
total_batches = len(points) // batch_size + (1 if len(points) % batch_size != 0 else 0)
uploaded_batches = 0
last_log_time = time()

# Use ThreadPoolExecutor to parallelize
with ThreadPoolExecutor(max_workers=6) as executor:  # Adjust `max_workers` for your CPU
    futures = [
        executor.submit(upload_batch, i, i + batch_size)
        for i in range(0, len(points), batch_size)
    ]

    # Monitor progress
    while not all(f.done() for f in futures):
        sleep(5)  # Check progress every 5 seconds
        completed = sum(1 for f in futures if f.done())
        if time() - last_log_time >= 30:  # Log progress every 30 seconds
            log_progress(f"{completed}/{total_batches} batches completed.")
            last_log_time = time()

# Final confirmation
log_progress("All batches uploaded!")
