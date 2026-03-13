import os
import cv2
import json
from paddlex import create_pipeline

# Initialize the PaddleX layout analysis pipeline
print("Loading PaddleX Universal Layout Analysis Pipeline...")
pipeline = create_pipeline(pipeline="universal_layout_analysis")

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

    # pipeline.predict returns a generator of results
    results = pipeline.predict(image_path)
    
    for res in results:
        # res['layout'] contains the detected blocks
        if 'layout' not in res:
            continue
            
        detected_blocks = res['layout']

        # Read image for cropping if we have blocks
        image = cv2.imread(image_path)
        if image is None:
            continue

        for i, block in enumerate(detected_blocks):
            # PaddleX block format: [x1, y1, x2, y2]
            bbox = block['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            block_type = block['label']

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