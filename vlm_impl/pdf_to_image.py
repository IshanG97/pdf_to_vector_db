import os
from pdf2image import convert_from_path

def pdf_to_image(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    image_paths = []  # To collect all image paths
    for pdf_file in os.listdir(input_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, pdf_file)
            # Convert PDF to images (one image per page)
            images = convert_from_path(pdf_path)
            for i, image in enumerate(images):
                # Replace spaces with underscores in the base name
                base_name = os.path.splitext(pdf_file)[0].replace(" ", "_")
                image_path = os.path.join(output_folder, f"{base_name}_page_{i + 1}.png")
                image.save(image_path, "PNG")
                image_paths.append(image_path)
    return image_paths

# Example usage
image_paths = pdf_to_image(input_folder="../input_pdfs", output_folder="../output_images")
print(f"PDFs converted into {len(image_paths)} images: {image_paths}")
