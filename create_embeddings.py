import os
import json
import torch
from PIL import Image
from typing import cast

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
    #device_map="cuda:0",  # or "mps" if on Apple Silicon
    device_map="mps",
).eval()

processor = ColPaliProcessor.from_pretrained(model_name)

# Process all images
image_embeddings_list = []
for img, path in zip(images, image_paths):
    processed_image = processor.process_images([img]).to(model.device)
    
    with torch.no_grad():
        # Extract embedding for this image
        image_embedding = model(**processed_image)[0].squeeze(0)  # Remove batch dimension if present
    
    # Append the embedding and corresponding image path
    image_embeddings_list.append({"path": path, "embedding": image_embedding.tolist()})

# Save the embeddings to a JSON file
output_file = "image_embeddings.json"
with open(output_file, "w") as f:
    json.dump(image_embeddings_list, f, indent=4)

print(f"Image embeddings saved to {output_file}.")

'''
# Process the inputs
batch_images = processor.process_images(images).to(model.device)
#batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    #query_embeddings = model(**batch_queries)

# Convert embeddings to a savable format (e.g., list of lists)
image_embeddings_list = [embedding.tolist() for embedding in image_embeddings]

'''

'''
# Inputs
images = [
    Image.new("RGB", (32, 32), color="white"),
    Image.new("RGB", (16, 16), color="black"),
]

'''

'''
# For scalability, we will need to batch process

batch_size = 10
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    batch_images = processor.process_images(batch).to(model.device)
    with torch.no_grad():
        batch_embeddings = model(**batch_images)
    # Save or process `batch_embeddings` here


'''

'''
# For passing queries directly, although this will likely be useless
queries = [
    "Is attention really all you need?",
    "Are Benjamin, Antoine, Merve, and Jo best friends?",
]

# This is if you are querying the output at the same time
#scores = processor.score_multi_vector(query_embeddings, image_embeddings)

'''