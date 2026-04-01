from unified_rag.retrieval.retriever import RetrievalEngine
from unified_rag.db.database import SessionLocal
import openai
from unified_rag.config import settings

# Configure OpenAI globally for the RAG prompt
openai.api_key = settings.openai_api_key

class RAGGenerator:
    """
    The core Multimodal Retrieval-Augmented Generation (RAG) engine.
    Operates in two strict modes:
      MODE 1 (SUMMARY): Short diagnostic summary only. No steps, no lists.
      MODE 2 (PROCEDURE): Structured JSON procedure only. No markdown.
    """
    def __init__(self):
        self.retriever = RetrievalEngine()
        
    def generate_response(self, query: str, manual_id: str, machine_id: str = None):
        """
        Executes the Multimodal RAG Pipeline with strict mode enforcement.
        """
        db = SessionLocal()
        try:
            # STAGE 1: SEMANTIC RETRIEVAL
            retrieved_data = self.retriever.retrieve(db, query, manual_id, machine_id)
            
            # STAGE 2: CONTEXT BUILDER
            text_context = ""
            pages = set()
            image_references = []
            
            for i, chunk in enumerate(retrieved_data["text_chunks"]):
                text_context += f"--- Manual Context {i+1} (Page {chunk.page}) ---\n{chunk.content}\n\n"
                if chunk.page is not None: pages.add(chunk.page)
            
            history_context = ""
            for i, fix in enumerate(retrieved_data.get("historical_fixes", [])):
                history_context += f"--- PREVIOUS FIX {i+1} ({fix.timestamp}) ---\nSummary: {fix.summary}\nOperator Actions: {fix.operator_fix}\n\n"

            for i, img in enumerate(retrieved_data["images"]):
                if img.path: image_references.append(img.path)
                text_context += f"--- Image Description {i+1} (Page {img.page}) ---\n{img.content}\n\n"
                
            # STAGE 3: MODE DETECTION
            is_procedure_request = "Generate full step-by-step repair procedure" in query or "FULL structured JSON repair procedure" in query
            
            if is_procedure_request:
                system_prompt = self._build_procedure_prompt(manual_id, text_context, history_context, image_references)
            else:
                system_prompt = self._build_summary_prompt(manual_id, text_context, history_context)
            
            try:
                user_content = f"Technician query: '{query}'"
                
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=3000,
                    temperature=0.1
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

    def _build_summary_prompt(self, manual_id: str, text_context: str, history_context: str) -> str:
        """
        MODE 1: Summary-only prompt. The LLM must produce a SHORT diagnostic overview.
        Absolutely NO procedures, NO numbered steps, NO bullet-point instructions.
        """
        return (
            f"You are an Industrial Diagnostic AI for: {manual_id}.\n\n"
            "YOUR TASK: Provide a SHORT diagnostic summary (3-5 sentences maximum).\n\n"
            "ABSOLUTE RULES:\n"
            "1. DO NOT provide ANY maintenance steps, repair instructions, or numbered procedures.\n"
            "2. DO NOT use bullet points or numbered lists to describe what to do.\n"
            "3. DO NOT say 'Step 1', 'Step 2', etc.\n"
            "4. DO NOT provide checklists or action items.\n"
            "5. ONLY describe WHAT the problem likely is and WHY based on sensor data.\n"
            "6. Keep your response under 5 sentences.\n"
            "7. You MUST end your response with EXACTLY this tag on its own line:\n"
            "   [SUGGESTION: Generate full step-by-step repair procedure]\n\n"
            "EXAMPLE OF A CORRECT RESPONSE:\n"
            "---\n"
            "Based on the sensor readings, the elevated temperature (176°C) combined with abnormal vibration patterns (0.82g) "
            "strongly suggests bearing degradation in the primary drive assembly. The motor current deviation of 5.1A indicates "
            "increased mechanical resistance, likely due to seal wear or lubrication breakdown. Immediate inspection is recommended "
            "before the fault progresses to a critical failure.\n\n"
            "[SUGGESTION: Generate full step-by-step repair procedure]\n"
            "---\n\n"
            f"MANUAL CONTEXT:\n{text_context}\n\n"
            f"HISTORICAL REPAIRS:\n{history_context}"
        )
    
    def _build_procedure_prompt(self, manual_id: str, text_context: str, history_context: str, image_references: list) -> str:
        """
        MODE 2: Structured JSON procedure prompt.
        The LLM must output ONLY the JSON wrapped in [PROCEDURE_START] and [PROCEDURE_END].
        No markdown, no explanatory text.
        """
        image_tags = "\n".join([f"  - [IMAGE_{i}] for image reference {i}" for i in range(len(image_references))])
        
        return (
            f"You are an Industrial Maintenance Procedure Generator for: {manual_id}.\n\n"
            "YOUR TASK: Generate a COMPLETE structured repair procedure in JSON format.\n\n"
            "ABSOLUTE RULES:\n"
            "1. Output ONLY the JSON wrapped between [PROCEDURE_START] and [PROCEDURE_END] tags.\n"
            "2. Do NOT write any text before or after the JSON block. No introductions. No summaries.\n"
            "3. The FIRST phase MUST be type 'safety' with lockout/tagout and PPE steps.\n"
            "4. Each task must have a unique 'id' (e.g. 's1', 's2', 't1', 't2', etc.).\n"
            "5. Mark safety/critical tasks with '\"critical\": true'.\n"
            "6. Include image references like [IMAGE_0], [IMAGE_1] in task text where relevant.\n"
            "7. Break down complex procedures into multiple subphases.\n"
            "8. Every single step from the manual should be included - be thorough.\n\n"
            f"AVAILABLE IMAGE TAGS:\n{image_tags}\n\n"
            "EXACT OUTPUT FORMAT (nothing else):\n"
            "[PROCEDURE_START]\n"
            '{"phases": [\n'
            '  {\n'
            '    "id": "safety_01",\n'
            '    "title": "⚠️ Safety & Lockout Protocols",\n'
            '    "type": "safety",\n'
            '    "subphases": [\n'
            '      {\n'
            '        "title": "Pre-Work Safety",\n'
            '        "tasks": [\n'
            '          {"id": "s1", "text": "Ensure machine is powered OFF and locked out per LOTO procedure", "critical": true},\n'
            '          {"id": "s2", "text": "Don required PPE: safety glasses, gloves, hearing protection", "critical": true}\n'
            '        ]\n'
            '      }\n'
            '    ]\n'
            '  },\n'
            '  {\n'
            '    "id": "maint_01",\n'
            '    "title": "🔧 Inspection & Repair",\n'
            '    "type": "maintenance",\n'
            '    "subphases": [\n'
            '      {\n'
            '        "title": "Visual Inspection",\n'
            '        "tasks": [\n'
            '          {"id": "t1", "text": "Inspect component for wear [IMAGE_0]", "critical": false}\n'
            '        ]\n'
            '      }\n'
            '    ]\n'
            '  }\n'
            ']}\n'
            "[PROCEDURE_END]\n\n"
            f"MANUAL CONTEXT:\n{text_context}\n\n"
            f"HISTORICAL REPAIRS:\n{history_context}"
        )
