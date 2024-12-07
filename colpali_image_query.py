from typing import cast
import json
import torch
from PIL import Image

from colpali_engine.models import ColPali, ColPaliProcessor

# Inputs
images = [
    Image.new("RGB", (32, 32), color="white"),
    Image.new("RGB", (16, 16), color="black"),
]

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
'''

# Model details

model_name = "vidore/colpali-v1.2"

model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    #device_map="cuda:0",  # or "mps" if on Apple Silicon
    device_map="mps",
).eval()

processor = ColPaliProcessor.from_pretrained(model_name)

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
#batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    #query_embeddings = model(**batch_queries)

# This is if you are querying the output at the same time
#scores = processor.score_multi_vector(query_embeddings, image_embeddings)

# Convert embeddings to a savable format (e.g., list of lists)
image_embeddings_list = [embedding.tolist() for embedding in image_embeddings]

# Save the embeddings to a JSON file
output_file = "image_embeddings.json"
with open(output_file, "w") as f:
    json.dump(image_embeddings_list, f)

print(f"Image embeddings saved to {output_file}.")
