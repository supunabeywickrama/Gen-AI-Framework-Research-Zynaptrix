import os
import cv2
import json
import numpy as np

# Monkeypatch numpy.bool for compatibility with older PaddleOCR
if not hasattr(np, 'bool'):
    np.bool = bool

# Try to import standard-imghdr if installed, or provide internal shim
try:
    import imghdr
except ImportError:
    # Minimal shim for imghdr.what if needed
    class ShimImghdr:
        def what(self, file, h=None):
            if hasattr(file, 'read'):
                header = file.read(32)
            elif isinstance(file, str):
                with open(file, 'rb') as f:
                    header = f.read(32)
            else:
                header = file[:32]
            
            if header.startswith(b'\xff\xd8'): return 'jpeg'
            if header.startswith(b'\x89PNG\r\n\x1a\n'): return 'png'
            if header.startswith(b'GIF87a') or header.startswith(b'GIF89a'): return 'gif'
            return None
    import sys
    sys.modules['imghdr'] = ShimImghdr()
    import imghdr

from paddleocr import PaddleStructure

# Initialize the PaddleOCR structure analysis engine
print("Loading PaddleOCR Layout Detection Model...")
# Use layout=True for layout analysis
engine = PaddleStructure(layout=True, show_log=False)

PAGES_FOLDER = "data/pages"
CROPPED_FOLDER = "data/cropped"
METADATA_FILE = "data/layout_regions.json"

if not os.path.exists(CROPPED_FOLDER):
    os.makedirs(CROPPED_FOLDER)

all_regions = []

print(f"Starting layout detection on images in {PAGES_FOLDER}...")

# Filter and sort files to process them in order
page_files = sorted([f for f in os.listdir(PAGES_FOLDER) if f.endswith(".png")])

for file in page_files:
    image_path = os.path.join(PAGES_FOLDER, file)
    print(f"Processing {file}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read {image_path}")
        continue

    # result is a list of dicts, each containing 'type', 'bbox', etc.
    # PaddleStructure(layout=True) returns a list of results
    result = engine(image)
    
    if not result:
        continue
        
    detected_blocks = result

    for i, block in enumerate(detected_blocks):
        # PaddleOCR bbox format: [x1, y1, x2, y2]
        bbox = block['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        block_type = block['type']

        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        cropped = image[y1:y2, x1:x2]
        
        # Create a unique filename for each region
        page_name = os.path.splitext(file)[0]
        region_filename = f"{page_name}_block_{i}.png"
        region_path = os.path.join(CROPPED_FOLDER, region_filename)
        
        cv2.imwrite(region_path, cropped)

        # Build metadata
        region_data = {
            "page": file,
            "region_id": i,
            "type": block_type,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }
        
        all_regions.append(region_data)

# Save all metadata to JSON
with open(METADATA_FILE, "w") as f:
    json.dump(all_regions, f, indent=4)

print(f"Layout detection complete. Saved {len(all_regions)} regions to {METADATA_FILE}")