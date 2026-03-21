import fitz  # PyMuPDF
import os
import easyocr
import camelot
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    pass

class DocumentParser:
    def __init__(self, upload_dir="data/uploads", output_dir="data/extracted", yolo_weights="models/yolov8_doclaynet.pt"):
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Initialize Layout Detection (YOLOv8 DocLayNet)
        print("Initializing YOLOv8 DocLayNet Layout Detection...")
        try:
            self.layout_model = YOLO(yolo_weights)
            print(f"Successfully loaded YOLOv8 weights from {yolo_weights}")
        except Exception as e:
            print(f"WARNING: Could not load YOLOv8 model from {yolo_weights}. Please download the weights! Error: {e}")
            self.layout_model = None

        # Initialize EasyOCR Reader
        print("Initializing EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=False) # Use gpu=True if CUDA is available
        
    def extract_text_with_ocr(self, image_path):
        """Uses EasyOCR to extract text from a specific image/region."""
        results = self.reader.readtext(image_path)
        text = " ".join([res[1] for res in results])
        return text

    def parse_pdf(self, file_path: str, manual_id: str):
        """
        Processes PDF:
        - Layout Detection YOLOv8 DocLayNet -> Text Blocks / Figures
        - Uses Trigram mapping back to the Page
        - EasyOCR fallback for dense regions
        - Camelot fallback for tables
        """
        print(f"📄 [Parser] Opening PDF: {file_path}")
        doc = fitz.open(file_path)
        parsed_data = []
        total_pages = len(doc)
        
        print(f"📖 [Parser] PDF has {total_pages} pages. Starting extraction...")
        
        for page_num in range(total_pages):
            print(f"   ∟ Processing Page {page_num + 1}/{total_pages}...")
            page = doc.load_page(page_num)
            
            # Use YOLOv8 AI Layout Detection if weights are installed
            if self.layout_model:
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Inference using YOLOv8
                results = self.layout_model(img, verbose=False)
                boxes = results[0].boxes
                names = self.layout_model.names
                
                img_index = 0
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    class_name = names[cls_id].lower()
                    
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Convert PIL coordinates back to PyMuPDF exact point scales
                    x_scale = page.rect.width / pix.width
                    y_scale = page.rect.height / pix.height
                    rect = fitz.Rect(x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale)
                    
                    # FIGURE / PICTURE: AI cleanly cropped
                    if "picture" in class_name or "figure" in class_name:
                        crop = img.crop((x1, y1, x2, y2))
                        
                        image_filename = f"{manual_id}_p{page_num+1}_img{img_index}.png"
                        image_path = os.path.join(self.output_dir, image_filename)
                        crop.save(image_path)
                        print(f"      [Image] Extracted figure to {image_path}")
                        
                        parsed_data.append({
                            "type": "image",
                            "path": image_path,
                            "page": page_num + 1
                        })
                        img_index += 1
                        
                    # TEXT BLOCKS: Accurately bound text
                    elif "text" in class_name or "title" in class_name or "list" in class_name:
                        text_content = page.get_text("text", clip=rect).strip()
                        if text_content:
                            parsed_data.append({
                                "type": "text",
                                "content": text_content,
                                "page": page_num + 1
                            })
            else:
                print(f"      [Warning] No YOLO model found, using basic text extraction for page {page_num+1}")
                # FALLBACK to PyMuPDF Heuristics if YOLO weights are missing
                blocks = page.get_text("blocks")
                for b in blocks:
                    if b[6] == 0:
                        text_content = b[4].strip()
                        if text_content:
                            parsed_data.append({"type": "text", "content": text_content, "page": page_num + 1})

        # Step 4: Extract Tables (Camelot)
        try:
            print(f"📊 [Parser] Extracting tables from {file_path} using Camelot (Lattice flavor)...")
            tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')
            print(f"✅ [Parser] Camelot found {len(tables)} potential tables.")
            for i, table in enumerate(tables):
                df = table.df
                table_str = df.to_csv(index=False) 
                parsed_data.append({
                    "type": "table",
                    "content": table_str,
                    "page": table.page
                })
                print(f"      [Table] Extracted table {i+1} from page {table.page}")
        except Exception as e:
            print(f"⚠️ [Parser] Camelot extraction issue: {e}")
            
        return parsed_data
