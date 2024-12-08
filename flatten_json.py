import json

# Load the embeddings from the JSON file
with open("image_embeddings.json", "r") as f:
    embeddings = json.load(f)

# Flatten the nested list for each embedding
for embedding in embeddings:
    embedding["embedding"] = [value for sublist in embedding["embedding"] for value in sublist]

# Save the flattened embeddings back to a file
output_file = "flattened_embeddings.json"
with open(output_file, "w") as f:
    json.dump(embeddings, f, indent=4)

print(f"Flattened embeddings saved to {output_file}.")
