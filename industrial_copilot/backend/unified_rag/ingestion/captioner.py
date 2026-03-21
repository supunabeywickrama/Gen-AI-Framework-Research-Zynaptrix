import base64
import openai
from unified_rag.config import settings

def encode_image(image_path: str):
    """Base64 encodes an image file to pass to the Vision LLM."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class ImageCaptioner:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        
    def generate_caption(self, image_path: str) -> str:
        """Uses OpenAI GPT-4o Vision API to describe the technical diagram/image in high detail."""
        print(f"      📸 [Vision] Sending image to OpenAI GPT-4o: {image_path}")
        try:
            base64_image = encode_image(image_path)
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "Describe this industrial/technical manual image in extremely high detail. Mention exact parts, labels, and what the diagram is demonstrating. This will be embedded as text to search for this image, so include keywords an engineer might use."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            caption = response.choices[0].message.content.strip()
            print(f"      ✨ [Vision] Successfully generated caption ({len(caption)} chars)")
            return caption
        except Exception as e:
            print(f"      ❌ [Vision] API FAILED for {image_path}: {e}")
            return "Image description unavailable."
