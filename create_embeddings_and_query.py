import os
from typing import cast
import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor

# Helper function to load images from a folder
def load_images_from_folder(folder):
    image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]
    images = [Image.open(image_path) for image_path in image_files]
    return images, image_files  # Return images and their paths for metadata

# Load images from the output_images folder
images, image_paths = load_images_from_folder("output_images")

# Model details
model_name = "vidore/colpali-v1.2"

model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="mps",  # or "mps" if on Apple Silicon
).eval()

processor = ColPaliProcessor.from_pretrained(model_name)

# Your queries
queries = [
    "Find a chart related to medical cost trends 2009 to 2025",
    "Is there a summary of the chart?"
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)  # Extract embeddings for images
    query_embeddings = model(**batch_queries)  # Extract embeddings for queries

# Compute similarity scores
scores = processor.score_multi_vector(query_embeddings, image_embeddings)

print(scores)

# Get top-k results for each query
top_k = 3
for query_idx, query_scores in enumerate(scores):
    sorted_indices = torch.argsort(query_scores, descending=True)
    print(f"Top {top_k} results for query {query_idx + 1}: '{queries[query_idx]}'")
    for i in range(top_k):
        index = sorted_indices[i].item()
        print(f"Image: {image_paths[index]} - Similarity: {query_scores[index].item()}")

