import os
import json
import torch
from colpali_engine.models import ColPali, ColPaliProcessor

# Load the embeddings from the JSON file
with open("image_embeddings.json", "r") as f:
    image_embeddings_data = json.load(f)

# Extract paths and embeddings
image_paths = [item["path"] for item in image_embeddings_data]
image_embeddings = [item["embedding"] for item in image_embeddings_data]

# Convert image embeddings to a tensor
image_embeddings_tensor = torch.tensor(image_embeddings, device="mps")
print(f"Image embeddings shape: {image_embeddings_tensor.shape}")

# Initialize ColPali model and processor
model_name = "vidore/colpali-v1.2"
model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="mps",  # Adjust for your hardware
).eval()
processor = ColPaliProcessor.from_pretrained(model_name)

# Input your text query
queries = [
    "Find me prices on IRA rebate policy?"
]

# Generate query embeddings
batch_queries = processor.process_queries(queries)
print(f"Batch queries type: {type(batch_queries)}")
print(f"Batch queries keys: {batch_queries.keys()}")  # Check the keys of the dictionary

# Move query data to the appropriate device
batch_queries = {key: value.to(model.device) for key, value in batch_queries.items()}

# Pass the processed queries to the model
with torch.no_grad():
    query_embeddings = model(**batch_queries)  # Extract query embeddings

print(f"Query embeddings shape: {query_embeddings.shape}")

# Compute similarity scores using ColPali's processor
scores = processor.score_multi_vector(query_embeddings, image_embeddings_tensor)

# Print scores
print(scores)

# Get top-k results for each query
top_k = 3
for query_idx, query_scores in enumerate(scores):
    sorted_indices = torch.argsort(query_scores, descending=True)
    print(f"Top {top_k} results for query {query_idx + 1}: '{queries[query_idx]}'")
    for i in range(top_k):
        index = sorted_indices[i].item()
        print(f"Image: {image_paths[index]} - Similarity: {query_scores[index].item()}")
