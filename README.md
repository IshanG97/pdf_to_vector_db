Converts PDFs into images - no need to patch the images as ColPALI does this for us
Parses the images into list of image embeddings in json format

NOTE: You need to install poppler for pdf_to_image.py
- macOS: brew install poppler
- Linux: sudo apt-get install poppler-utils
- Windows: look at https://poppler.freedesktop.org/

TO DO:
- batch processing in create_embeddings.py