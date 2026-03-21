class ContextualChunker:
    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_data(self, parsed_data: list, manual_id: str):
        """
        Implements sliding window chunking for text and tables.
        Retains metadata and now handles Vision-generated content for images.
        """
        chunks = []
        for item in parsed_data:
            if item["type"] in ["text", "table"]:
                text = item["content"]
                words = text.split()
                
                if len(words) <= self.chunk_size:
                    chunks.append({
                        "manual_id": manual_id,
                        "type": item["type"],
                        "content": text,
                        "page": item["page"]
                    })
                else:
                    for i in range(0, len(words), self.chunk_size - self.overlap):
                        chunk_words = words[i:i + self.chunk_size]
                        chunk_text = " ".join(chunk_words)
                        chunks.append({
                            "manual_id": manual_id,
                            "type": item["type"],
                            "content": chunk_text,
                            "page": item["page"]
                        })
            elif item["type"] == "image":
                chunks.append({
                    "manual_id": manual_id,
                    "type": "image",
                    "path": item["path"],
                    "content": item.get("content", ""), # Specifically preserve Vision Captions
                    "page": item["page"]
                })
                
        return chunks
