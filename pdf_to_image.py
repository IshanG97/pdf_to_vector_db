from pdf2image import convert_from_path

def pdf_to_images(pdf_path, output_folder="output_images"):
    # Convert PDF to images (one image per page)
    images = convert_from_path(pdf_path)
    image_paths = []
    for i, image in enumerate(images):
        image_path = f"{output_folder}/page_{i + 1}.png"
        image.save(image_path, "PNG")
        image_paths.append(image_path)
    return image_paths

# Example usage
pdf_path = "example.pdf"
image_paths = pdf_to_images(pdf_path)
print(f"PDF converted into {len(image_paths)} images: {image_paths}")
