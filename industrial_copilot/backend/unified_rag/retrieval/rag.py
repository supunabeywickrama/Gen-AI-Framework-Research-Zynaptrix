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
        
    def generate_response(self, query: str, manual_id: str, machine_id: str = None):
        """
        Executes the three-stage Multimodal RAG Pipeline:
        1. Semantic Retrieval (Manual + History)
        2. Visual Context Ingestion
        3. Logic Consolidation: Generates a step-by-step repair guide.
        """
        db = SessionLocal()
        try:
            # STAGE 1: SEMANTIC RETRIEVAL
            retrieved_data = self.retriever.retrieve(db, query, manual_id, machine_id)
            
            # STAGE 2: CONTEXT BUILDER
            text_context = ""
            pages = set()
            image_references = []
            
            # Aggregate Text Chunks
            for i, chunk in enumerate(retrieved_data["text_chunks"]):
                text_context += f"--- Manual Context {i+1} (Page {chunk.page}) ---\n{chunk.content}\n\n"
                if chunk.page is not None: pages.add(chunk.page)
            
            # Aggregate Historical Field Fixes (PRIORITY CONTEXT)
            history_context = ""
            for i, fix in enumerate(retrieved_data.get("historical_fixes", [])):
                history_context += f"--- PREVIOUS REAL-WORLD FIX {i+1} ({fix.timestamp}) ---\nSummary: {fix.summary}\nOperator Actions: {fix.operator_fix}\n\n"

            # Aggregate Technical Diagrams
            for i, img in enumerate(retrieved_data["images"]):
                if img.path: image_references.append(img.path)
                text_context += f"--- Image Description {i+1} (Page {img.page}) ---\n{img.content}\n\n"
                
            # STAGE 3: CONTEXTUAL REASONING
            image_list_str = "\n".join([f"- [IMAGE_{i}]: {img.content}" for i, img in enumerate(retrieved_data["images"])])
            
            system_prompt = (
                f"You are a Senior Industrial Systems Engineer specializing in: {manual_id}.\n"
                "Your objective is to provide precise, technically accurate answers using both the OFFICIAL MANUAL and HISTORICAL FIELD FIXES.\n\n"
                "### KNOWLEDGE SOURCES:\n"
                "1. **OFFICIAL MANUAL**: Theoretical ground truth.\n"
                "2. **HISTORICAL FIELD FIXES**: Practical solutions previously implemented by operators on this machine. If these contradict the manual but were successful, mention them as 'field-proven alternatives'.\n\n"
                "### INTERACTION PROTOCOL:\n"
                "1. **SCOPE**: Answer the specific query accurately.\n"
                "2. **SEMANTIC INTERLEAVING**: Embed [IMAGE_N] where helpful.\n"
                "3. **PREMIUM SUGGESTIONS**: End with exactly 3 follow-up queries formatted as [SUGGESTION: query].\n\n"
                f"### PREVIOUS SUCCESSFUL FIXES (HISTORICAL CONTEXT):\n{history_context if history_context else 'No previous records for this fault.'}\n\n"
                f"### OFFICIAL MANUAL CONTEXT:\n{text_context}"
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
