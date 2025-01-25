import os
import fitz
from PyPDF2 import PdfReader
import spacy

# Load spacy model once at module level
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
# Disable unnecessary components for better performance

def extract_metadata(file_path):
    """Extract metadata and content from PDF file."""
    doc = fitz.open(file_path)
    pdf_reader = PdfReader(file_path)

    metadata = {
        "filename": os.path.basename(file_path),
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
        "subject": doc.metadata.get("subject", ""),
        "keywords": doc.metadata.get("keywords", ""),
        "creator": doc.metadata.get("creator", ""),
        "producer": doc.metadata.get("producer", ""),
        "creation_date": doc.metadata.get("creationDate", ""),
        "modification_date": doc.metadata.get("modDate", ""),
        "page_count": len(doc),
        "custom_metadata": pdf_reader.metadata,
        "pages": [],
    }

    # Process each page for additional metadata
    for page_num, page in enumerate(doc, start=1):
        page_data = {
            "page_number": page_num,
            "text": page.get_text(),
            "dimensions": {"width": page.rect.width, "height": page.rect.height},
            "images": [],
            "sentences": [],
        }

        # Extract images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_info = {
                "image_index": img_index,
                "image_bytes": base_image["image"],
                "width": base_image.get("width"),
                "height": base_image.get("height"),
            }
            page_data["images"].append(image_info)

        # Extract sentences using spacy
        doc_text = nlp(page_data["text"])
        sentences = [sent.text.strip() for sent in doc_text.sents if sent.text.strip()]
        page_data["sentences"] = sentences
        metadata["pages"].append(page_data)

    return metadata