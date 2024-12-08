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