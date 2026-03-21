from unified_rag.retrieval.retriever import RetrievalEngine
from unified_rag.db.database import SessionLocal
import openai
from unified_rag.config import settings

# Configure OpenAI globally for the RAG prompt
openai.api_key = settings.openai_api_key

class RAGGenerator:
    def __init__(self):
        self.retriever = RetrievalEngine()
        
    def generate_response(self, query: str, manual_id: str):
        """
        Executes the full RAG sequence: Retrieve -> Build Context -> Generate LLM Response.
        """
        db = SessionLocal()
        try:
            # 1. Retrieve
            retrieved_data = self.retriever.retrieve(db, query, manual_id)
            
            # 2. Context Builder
            text_context = ""
            pages = set()
            image_references = []
            
            for i, chunk in enumerate(retrieved_data["text_chunks"]):
                text_context += f"--- Content Block {i+1} (Page {chunk.page}, Type: {chunk.type}) ---\n{chunk.content}\n\n"
                if chunk.page is not None:
                    pages.add(chunk.page)
                    
            for i, img in enumerate(retrieved_data["images"]):
                if img.path:
                    image_references.append(img.path)
                if img.page is not None:
                    pages.add(img.page)
                
                # INJECT VISION CAPTION INTO RAG CONTEXT
                text_context += f"--- Image Description {i+1} (Page {img.page}, Type: {img.type}) ---\n{img.content}\n\n"
                
            # 3. RAG Response Generation using LLM
            # We use gpt-4o for complex reasoning over technical context
            system_prompt = (
                "You are an Industrial AI Troubleshooting Copilot. "
                "Use the provided context extracted from industrial manuals to answer the user's query.\n"
                "Provide step-by-step diagnostic or repair strategies if asked.\n"
                "If the context does not contain the answer, state that you do not know.\n\n"
                f"# CONTEXT:\n{text_context}"
            )
            
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=700,
                    temperature=0.2 # Lower temperature for factual accuracy
                )
                answer = response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI API call failed: {e}")
                answer = "Error generating response from LLM."
            
            return {
                "answer": answer,
                "images": image_references,
                "pages": sorted(list(pages))
            }
        except Exception as e:
            print(f"RAG formulation failed: {e}")
            raise e
        finally:
            db.close()
