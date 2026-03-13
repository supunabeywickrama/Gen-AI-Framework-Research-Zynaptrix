import os
import cv2
import json
import urllib.request
from ultralytics import YOLO

PAGES_FOLDER = "data/pages"
CROPPED_FOLDER = "data/cropped"
MODELS_FOLDER = "data/models"
METADATA_FILE = "data/layout_regions.json"

# Download a pre-trained YOLOv8 DocLayNet model from Hugging Face
MODEL_URL = "https://huggingface.co/hantian/yolo-doclaynet/resolve/main/yolov8s-doclaynet.pt"
MODEL_PATH = os.path.join(MODELS_FOLDER, "yolov8s-doclaynet.pt")

if not os.path.exists(CROPPED_FOLDER):
    os.makedirs(CROPPED_FOLDER)
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)

if not os.path.exists(MODEL_PATH):
    print(f"Downloading YOLOv8 DocLayNet model from {MODEL_URL}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download model: {e}")
        exit(1)

print("Loading YOLO Model...")
# Load the downloaded model weights
model = YOLO(MODEL_PATH)

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

    # Run detection
    results = model(image, verbose=False)
    
    # Process bounding boxes
    for i, box in enumerate(results[0].boxes):
        # Coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        block_type = model.names[cls_id]

        # Ignore very low confidence detections
        if conf < 0.25:
            continue

        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        cropped = image[y1:y2, x1:x2]
        
        # Create a unique filename for each region
        page_name = os.path.splitext(file)[0]
        region_filename = f"{page_name}_region_{i}_{block_type}.png"
        region_path = os.path.join(CROPPED_FOLDER, region_filename)
        
        cv2.imwrite(region_path, cropped)

        # Build metadata
        region_data = {
            "page": file,
            "region_id": i,
            "type": block_type,
            "confidence": conf,
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