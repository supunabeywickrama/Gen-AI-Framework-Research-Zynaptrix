from pdf2image import convert_from_path
import os

PDF_PATH = "data/manuals/esp32_datasheet_en.pdf"
OUTPUT_FOLDER = "data/pages"

def convert_pdf_to_images():
    print("Starting PDF conversion...")

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    pages = convert_from_path(PDF_PATH, dpi=300)

    print(f"Total pages found: {len(pages)}")

    for i, page in enumerate(pages):

        image_path = os.path.join(
            OUTPUT_FOLDER,
            f"page_{i+1}.png"
        )

        page.save(image_path, "PNG")

        print(f"Saved {image_path}")

    print("PDF conversion complete.")

if __name__ == "__main__":
    convert_pdf_to_images()