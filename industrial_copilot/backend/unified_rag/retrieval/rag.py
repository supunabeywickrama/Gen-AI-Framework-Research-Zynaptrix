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
            
            # Prepare image inventory for the LLM
            image_list_str = "\n".join([f"- [IMAGE_{i}]: {img.content} (Type: {img.type})" for i, img in enumerate(retrieved_data["images"])])
            
            system_prompt = (
                "You are a Senior Industrial Systems Engineer specializing in the Zynaptrix-9000 Turbo Pump.\n"
                "Your objective is to generate a comprehensive, human-readable repair procedure based ONLY on the provided manual context.\n\n"
                "### CRITICAL FORMATTING RULES:\n"
                "1. **NO INTRO/OUTRO**: Do not start with 'Based on the context' or 'RAG Extract' or 'Here is the procedure'. Start directly with the first heading.\n"
                "2. Use markdown headers (##, ###) for sections.\n"
                "3. **INTERLEAVE IMAGES (MANDATORY)**: You MUST include the technical diagrams listed below to assist the operator. "
                "Insert the tag [IMAGE_N] at the EXACT point in your instructions where that visual aid is most relevant. "
                "Example: 'Locate the primary discharge port as shown in [IMAGE_2].'\n"
                "4. Ensure every retrieved image is referenced at least once if it is relevant to the anomaly.\n\n"
                f"### AVAILABLE TECHNICAL DIAGRAMS:\n{image_list_str}\n\n"
                f"### OFFICIAL MANUAL CONTEXT:\n{text_context}"
            )
            
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Query: {query}\n\nPlease generate a full maintenance procedure."}
                    ],
                    max_tokens=1500,
                    temperature=0.2
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
