import os
from openai import OpenAI
from unified_rag.config import settings

class MultimodalEmbedder:
    def __init__(self):
        # Initialize OpenAI Client (Unified Embedding Space)
        api_key = settings.openai_api_key
        if not api_key:
            print("WARNING: OPENAI_API_KEY is not set.")
        self.openai_client = OpenAI(api_key=api_key) if api_key else None
        self.text_model = "text-embedding-3-small" # 1536 dimensions
        
    def embed_text(self, text: str) -> list[float]:
        """Embeds text, tables, and vision-generated image captions using OpenAI."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Check API key.")
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.text_model
        )
        return response.data[0].embedding

embedder = MultimodalEmbedder()
