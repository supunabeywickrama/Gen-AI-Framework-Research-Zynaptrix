import layoutparser as lp
import cv2
import os
import json

# Load pre-trained document layout model
print("Loading Layout Detection Model...")
model = lp.Detectron2LayoutModel(
    "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
    label_map={
        0: "Text",
        1: "Title",
        2: "List",
        3: "Table",
        4: "Figure"
    }
)

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

    layout = model.detect(image)

    for i, block in enumerate(layout):
        # Crop region
        x1 = int(block.block.x_1)
        y1 = int(block.block.y_1)
        x2 = int(block.block.x_2)
        y2 = int(block.block.y_2)

        cropped = image[y1:y2, x1:x2]
        
        # Create a unique filename for each region
        page_name = os.path.splitext(file)[0]
        region_filename = f"{page_name}_region_{i}.png"
        region_path = os.path.join(CROPPED_FOLDER, region_filename)
        
        cv2.imwrite(region_path, cropped)

        # Build metadata
        region_data = {
            "page": file,
            "region_id": i,
            "type": block.type,
            "x1": x1,
            "y1": y1,
            "x2": x1, # User example had x2 as x2, but let's be safe
            "y2": y2
        }
        # Correcting x2 assignment from my thought
        region_data["x2"] = x2
        
        all_regions.append(region_data)

# Save all metadata to JSON
with open(METADATA_FILE, "w") as f:
    json.dump(all_regions, f, indent=4)

print(f"Layout detection complete. Saved {len(all_regions)} regions to {METADATA_FILE}")