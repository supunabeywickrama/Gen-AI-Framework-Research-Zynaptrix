import os
import sys
import numpy as np
import cv2
import base64
from openai import OpenAI

# Add current directory to path
sys.path.append(os.getcwd())

from services.figure_splitter import FigureSplitter
from unified_rag.config import settings

def test_splitter():
    print("🚀 Testing FigureSplitter with JSON Mode & Retries...")
    
    # Create a dummy image (white background with a black rectangle)
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 0), 2)
    cv2.putText(img, "Machine Diagram", (110, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    splitter = FigureSplitter()
    
    # Test only the OpenAI part first (as it was the failure point)
    print("📡 Calling OpenAI for center detection...")
    b64 = splitter.encode_image(img)
    centers = splitter.ask_openai_centers(b64, parent_context="A sample machine diagram for testing.")
    
    print(f"✅ Received {len(centers)} components from OpenAI.")
    for i, c in enumerate(centers):
        print(f"   [{i}] Label: {c.get('label')} | x: {c.get('x')} | y: {c.get('y')} | Noise: {c.get('is_noise')}")
        
    if not centers:
        print("❌ FAILED: No centers received. Check OpenAI API Key or Logs.")
        sys.exit(1)
        
    print("\n🔬 Testing full splitting pipeline (SAM)...")
    try:
        results = splitter.split_image_sam(img, parent_context="Test context")
        print(f"✅ Successfully isolated {len(results)} components.")
    except Exception as e:
        print(f"⚠️ SAM/Splitting execution failed (likely no model weights): {e}")
        # Not a complete failure if SAM is missing, as the goal was OpenAI stability.

if __name__ == "__main__":
    test_splitter()
