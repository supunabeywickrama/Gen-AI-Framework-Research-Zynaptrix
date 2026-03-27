from unified_rag.retrieval.retriever import RetrievalEngine
from unified_rag.db.database import SessionLocal
import openai
from unified_rag.config import settings

# Configure OpenAI globally for the RAG prompt
openai.api_key = settings.openai_api_key

class RAGGenerator:
    """
    The core Multimodal Retrieval-Augmented Generation (RAG) engine.
    This component bridges the gap between structured vector search and unstructured 
    visual reasoning using GPT-4o Vision.
    """
    def __init__(self):
        self.retriever = RetrievalEngine()
        
    def generate_response(self, query: str, manual_id: str):
        """
        Executes the three-stage Multimodal RAG Pipeline:
        1. Semantic Retrieval (Vector DB): Fetches relevant text chunks and images.
        2. Visual Context Ingestion: Injects GPT-4o Vision captions into the context window.
        3. Logic Consolidation: Generates a step-by-step repair guide with interleaved visuals.
        """
        db = SessionLocal()
        try:
            # STAGE 1: SEMANTIC RETRIEVAL
            # Fetches the top-k text chunks and associated technical figures.
            retrieved_data = self.retriever.retrieve(db, query, manual_id)
            
            # STAGE 2: CONTEXT BUILDER (Multimodal Synthesis)
            text_context = ""
            pages = set()
            image_references = []
            
            # Aggregate Text Chunks with Metadata
            for i, chunk in enumerate(retrieved_data["text_chunks"]):
                text_context += f"--- Content Block {i+1} (Page {chunk.page}, Type: {chunk.type}) ---\n{chunk.content}\n\n"
                if chunk.page is not None:
                    pages.add(chunk.page)
            
            # Aggregate Technical Diagrams with GPT-4o Vision Captions
            for i, img in enumerate(retrieved_data["images"]):
                if img.path:
                    image_references.append(img.path)
                if img.page is not None:
                    pages.add(img.page)
                
                # IMPORTANT: We inject the VISION CAPTION as text so the LLM knows what's in the diagram.
                # This allows the LLM to 'see' the technical details during the strategy phase.
                text_context += f"--- Image Description {i+1} (Page {img.page}, Type: {img.type}) ---\n{img.content}\n\n"
                
            # STAGE 3: CONTEXTUAL REASONING (OpenAI GPT-4o)
            # We use gpt-4o for its superior technical reasoning and multimodal awareness.
            
            # Provide the LLM with an 'Inventory' of available diagrams for interleaving.
            image_list_str = "\n".join([f"- [IMAGE_{i}]: {img.content} (Type: {img.type})" for i, img in enumerate(retrieved_data["images"])])
            
            system_prompt = (
                f"You are a Senior Industrial Systems Engineer specializing in the technical manual for: {manual_id}.\n"
                "Your objective is to provide precise, technically accurate answers based ONLY on the provided manual context.\n\n"
                "### INTERACTION PROTOCOL:\n"
                "1. **SCOPE**: If the user asks a specific question, answer ONLY that question. Do not provide a full repair procedure unless asked.\n"
                "2. **TECHNICAL HIERARCHY**: Use ## and ### for sections where appropriate.\n"
                "3. **SEMANTIC INTERLEAVING**: Embed the tag [IMAGE_N] at the EXACT point where that technical diagram aids the explanation.\n"
                "4. **PREMIUM SUGGESTIONS (MANDATORY)**: At the very end of your response, provide exactly 3 follow-up suggestions for the operator. "
                "Format them strictly as: [SUGGESTION: Type specific follow-up query here]\n\n"
                f"### AVAILABLE TECHNICAL DIAGRAMS (INVENTORY):\n{image_list_str}\n\n"
                f"### OFFICIAL MANUAL CONTEXT (TEXT + VISION CAPTIONS):\n{text_context}"
            )
            
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Query: {query}\n\nPlease generate the maintenance procedure as a Senior Engineer."}
                    ],
                    max_tokens=1500,
                    temperature=0.2 # Low temperature for high technical precision
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
