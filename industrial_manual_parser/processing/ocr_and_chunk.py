import os
import json
import easyocr

# Use absolute or relative paths with os.path structure matching project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CROPPED_FOLDER = os.path.join(DATA_DIR, "cropped")
METADATA_FILE = os.path.join(DATA_DIR, "layout_regions.json")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")

def sliding_chunks(text, chunk_size=1000, overlap=250):
    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(text), step):
        end = start + chunk_size
        chunk = text[start:end]

        if len(chunk) < 100:
            break

        chunks.append({
            "start": start,
            "end": end,
            "text": chunk
        })

    return chunks

def extract_text():
    print(f"Loading layout metadata from {METADATA_FILE}...")
    with open(METADATA_FILE, "r") as f:
        regions = json.load(f)

    TEXT_CLASSES = [
        "Text",
        "Title",
        "Section-header",
        "Caption",
        "List-item"
    ]

    text_regions = [r for r in regions if r.get("type") in TEXT_CLASSES]
    
    # Sort regions properly by page number (page_X.png) and then roughly by y1 coordinate 
    def get_page_num(page_str):
        # Extract number from 'page_X.png'
        try:
            return int(page_str.split('_')[1].split('.')[0])
        except:
            return 0
            
    text_regions.sort(key=lambda r: (get_page_num(r["page"]), r["y1"], r["x1"]))

    print(f"Initializing EasyOCR reader. Found {len(text_regions)} text regions to process...")
    # By default, gpu=True if CUDA is available, otherwise falls back to CPU.
    reader = easyocr.Reader(['en']) 

    full_text = ""
    for idx, region in enumerate(text_regions):
        page_name = os.path.splitext(region["page"])[0]
        region_id = region["region_id"]
        block_type = region["type"]
        region_filename = f"{page_name}_region_{region_id}_{block_type}.png"
        img_path = os.path.join(CROPPED_FOLDER, region_filename)

        if not os.path.exists(img_path):
            print(f"Warning: Image not found {img_path}")
            continue

        if idx % 50 == 0:
            print(f"Processing region {idx}/{len(text_regions)}...")

        result = reader.readtext(img_path, detail=0)
        extracted = " ".join(result)
        if extracted.strip():
            full_text += extracted + "\n"

    print("Finished extracting text from all regions.")
    
    print("Generating sliding window chunks...")
    chunks = sliding_chunks(full_text, chunk_size=1000, overlap=250)
    
    chunk_data = []
    source_name = "motor_manual"
    
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "chunk_id": i + 1,
            "start_char": chunk["start"],
            "end_char": chunk["end"],
            "text": chunk["text"],
            "source": source_name
        })

    print(f"Writing {len(chunk_data)} chunks to {CHUNKS_FILE}...")
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=4)
        
    print("Phase 4 Output Generation Complete.")

if __name__ == "__main__":
    extract_text()
