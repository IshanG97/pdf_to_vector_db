from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor

# Initialize ColPali model and processor
model_name = "vidore/colpali-v1.2"
model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # Adjust for your hardware
).eval()
processor = ColPaliProcessor.from_pretrained(model_name)

# Load images from the PDF
image_paths = pdf_to_images("example.pdf")
images = [Image.open(image_path) for image_path in image_paths]

# Process images with ColPali
batch_images = processor.process_images(images).to(model.device)

with torch.no_grad():
    image_embeddings = model(**batch_images)

# (Optional) Store image embeddings for retrieval
print("Image embeddings generated.")
