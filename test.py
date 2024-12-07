from PIL import Image
import os

# Helper function to load images
def load_images_from_folder(folder):
    image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]
    images = [Image.open(image_file) for image_file in image_files]  # Load as PIL.Image
    return images

images = load_images_from_folder("output_images")

for img in images:
    print(type(img))  # Should print <class 'PIL.Image.Image'>

# Convert images to RGB mode (if not already)
images = [img.convert("RGB") for img in images]

# Verify the type and mode
for img in images:
    print(f"Type: {type(img)}, Mode: {img.mode}")