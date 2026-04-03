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
        
    def generate_caption(self, image_path: str, metadata: dict = None) -> str:
        """
        Uses OpenAI GPT-4o Vision API to describe the image with full document context.
        """
        metadata = metadata or {}
        page = metadata.get("page", "Unknown")
        section = metadata.get("section", "Unknown Section")
        label = metadata.get("label", "Diagram")
        parent_ctx = metadata.get("parent_context", "")
        
        # Build a hyper-contextualized prompt
        context_str = f"This image is a technical illustration labeled '{label}' on Page {page} of the manual."
        if section != "Unknown Section":
            context_str += f" It is located within the section: '{section}'."
        if parent_ctx:
            context_str += f" Context: {parent_ctx}."

        print(f"      📸 [Vision] Sending image to GPT-4o with context: {label} (Page {page})")
        
        prompt = (
            f"You are a Senior Industrial Systems Engineer. {context_str}\n\n"
            "INSTRUCTIONS:\n"
            "1. Describe this specific technical component in extremely high detail.\n"
            "2. Explain its function and relationship to the surrounding assembly mentioned in the context.\n"
            "3. Identify any labels, bolts, connectors, or part numbers visible.\n"
            "4. Use professional engineering terminology. This description will be used for RAG retrieval, "
            "so include keywords that a technician would use when troubleshooting this specific part."
        )

        try:
            base64_image = encode_image(image_path)
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
                        ]
                    }
                ],
                max_tokens=500
            )
            caption = response.choices[0].message.content.strip()
            return f"### {label} (Context: {section})\n\n{caption}"
        except Exception as e:
            print(f"      ❌ [Vision] API FAILED for {image_path}: {e}")
            return f"Technical diagram '{label}' on Page {page}. Description unavailable."
