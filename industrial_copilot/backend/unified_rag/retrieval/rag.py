from enum import Enum
from unified_rag.retrieval.retriever import RetrievalEngine
from unified_rag.db.database import SessionLocal
import openai
from unified_rag.config import settings

# Configure OpenAI globally for the RAG prompt
openai.api_key = settings.openai_api_key

class RAGMode(Enum):
    SUMMARY = "summary"
    PROCEDURE = "procedure"
    CLARIFICATION = "clarification"
    EVALUATION = "evaluation"
    CONVERSATIONAL_WIZARD = "conversational_wizard"

class RAGGenerator:
    """
    The core Multimodal Retrieval-Augmented Generation (RAG) engine.
    Operates in four strict modes:
      MODE 1 (SUMMARY): Short diagnostic summary only. No steps, no lists.
      MODE 2 (PROCEDURE): Structured JSON procedure only. No markdown.
      MODE 3 (CLARIFICATION): Detailed, non-technical drill-down for a specific maintenance step.
      MODE 4 (EVALUATION): Evaluates technician feedback against manual requirements.
    """
    def __init__(self):
        self.retriever = RetrievalEngine()
        
    def generate_response(self, query: str, manual_id: str, machine_id: str, mode: RAGMode = RAGMode.SUMMARY, chat_history: str = "") -> dict:
        """
        Executes the Multimodal RAG Pipeline with strict mode enforcement.
        Now supports passing the active chat history for 'Conversational Wizard' mode.
        """
        db = None
        retrieved_data = {"text_chunks": [], "images": [], "historical_fixes": []}
        
        try:
            # STAGE 1: SEMANTIC RETRIEVAL (Fault Tolerant)
            try:
                db = SessionLocal()
                retrieved_data = self.retriever.retrieve(db, query, manual_id, machine_id)
            except Exception as e:
                print(f"RAG_DB_ERROR: {e}. Falling back to manual-only context.")
            finally:
                if db:
                    db.close()
            
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
                
            # STAGE 3: MODE SELECTION
            if mode == RAGMode.PROCEDURE:
                system_prompt = self._build_procedure_prompt(manual_id, text_context, history_context, image_references)
            elif mode == RAGMode.CLARIFICATION:
                system_prompt = self._build_clarification_prompt(manual_id, query, text_context, image_references)
            elif mode == RAGMode.EVALUATION:
                system_prompt = self._build_evaluation_prompt(manual_id, query, text_context, image_references)
            elif mode == RAGMode.CONVERSATIONAL_WIZARD:
                system_prompt = self._build_conversational_wizard_prompt(
                    manual_id, 
                    query, 
                    text_context, 
                    image_references,
                    chat_history
                )
            else:
                system_prompt = self._build_summary_prompt(manual_id, text_context, history_context)
            
            # STAGE 4: LLM INFERENCE
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
            "7. You MUST end your response with EXACTLY this tag on its own line to trigger the repair wizard:\n"
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

    def _build_clarification_prompt(self, manual_id: str, step_text: str, text_context: str, image_references: list) -> str:
        """
        MODE 3: Step Clarification Prompt.
        """
        image_tags = "\n".join([f"  - [IMAGE_{i}] for image reference {i}" for i in range(len(image_references))])
        
        return (
            f"You are a Technical Mentor for: {manual_id}.\n\n"
            f"YOUR TASK: Provide a detailed, easy-to-understand explanation for the maintenance task described below.\n"
            f"If the input looks like a troubleshooting advice, focus on explaining the ORIGINAL task context found in the manual.\n\n"
            f"TASK TO EXPLAIN: '{step_text}'\n\n"
            "CONTEXT RULES:\n"
            "1. Explain the task to someone with NO technical background. Avoid jargon.\n"
            "2. Use bullet points for sub-tasks if necessary.\n"
            "3. Reference technical diagrams using the available image tags if they help illustrate THIS specific task.\n"
            "4. Include safety warnings if the task is dangerous.\n"
            "5. Explain WHAT to do, why it matters (WHY), and HOW to do it correctly. Keep your explanation to 2-3 detailed, warm sentences.\n"
            "6. Your response should be encouraging and supportive.\n\n"
            f"AVAILABLE IMAGE TAGS:\n{image_tags}\n\n"
            f"MANUAL CONTEXT TO USE AS SOURCE OF TRUTH:\n{text_context}"
        )

    def _build_evaluation_prompt(self, manual_id: str, user_feedback: str, text_context: str, image_references: list) -> str:
        """
        MODE 4: Step Evaluation Prompt.
        Determines if a user's feedback indicates task completion or failure.
        """
        return (
            f"You are a Quality Assurance Supervisor for: {manual_id}.\n\n"
            f"A technician has provided the following feedback on their progress:\n"
            f"'{user_feedback}'\n\n"
            "YOUR JOB:\n"
            "1. Evaluate if the technician has successfully completed the task described in the manual context.\n"
            "2. If they have completed it, start your response with '[STEP_COMPLETE]'. Provide a short confirmation message.\n"
            "3. If they are having trouble, or their comment suggests a failure, start your response with '[STEP_NEED_HELP]'. Provide a helpful troubleshooting tip based ONLY on the manual context.\n"
            "4. Be concise and professional.\n\n"
            f"MANUAL SOURCE OF TRUTH:\n{text_context}"
        )

    def _build_conversational_wizard_prompt(self, manual_id: str, user_query: str, text_context: str, image_references: list, history: str) -> str:
        """
        MODE 5: Conversational Wizard Prompt.
        The core "Brain" of the intelligent maintenance flow.
        """
        image_tags = "\n".join([f"  - [IMAGE_{i}] for image reference {i}" for i in range(len(image_references))])
        
        return (
            f"You are an Intelligent Maintenance Mentor for: {manual_id}.\n\n"
            "YOUR OBJECTIVE:\n"
            "Guide the operator through the repair process conversationally. Instead of a rigid list, you own the flow.\n\n"
            "CURRENT CONTEXT:\n"
            f"1. CHAT HISTORY (Full record of actions and instructions so far):\n{history}\n"
            f"2. OPERATOR'S RECENT INPUT: '{user_query}'\n"
            f"3. MANUAL SOURCE OF TRUTH (Procedures and diagrams):\n{text_context}\n\n"
            "YOUR MISSION CRITICAL TASK:\n"
            "1. READ THE HISTORY FIRST: Scan the memory to see exactly what has happened. If the history is empty, start from the beginning (Safety).\n"
            "2. PHASE DETECTION:\n"
            "   - If history doesn't show a completed '[PHASE: Safety & Lockout]', you MUST start there. This is non-negotiable.\n"
            "   - If Safety is done, look for the current Repair phase in the manual context and history to determine the exact next instruction.\n"
            "3. ADAPT TO OPERATOR INTENT:\n"
            "   - IF input contains 'NEED_HELP' or they seem stuck: DO NOT ADVANCE. Provide a detailed 'How-To' or an alternative way from the manual context to help them overcome the hurdle.\n"
            "   - IF input contains 'NEED_DETAIL': Provide a non-technical breakdown of the CURRENT step. Explain 'What', 'Why', and 'How' with warm English.\n"
            "   - IF input contains 'CONFIRM_DONE' or they say they are finished: You MUST find the NEXT logical instruction in the 'MANUAL SOURCE OF TRUTH' and provide it immediately. Do NOT ask 'what's next' - YOU are the guide. Provide the step details.\n"
            "4. RESPONSE STYLE: 2-3 detailed sentences. Be warm, supportive, and clear. Maintain continuity - if they just finished 'Step A', your response MUST be 'Step B'.\n\n"
            "OUTPUT FORMAT:\n"
            "- Start with '[PHASE: <Name>]' (e.g. Safety & Lockout, Disassembly, Component Repair).\n"
            "- If the entire procedure is finished (after the final verification step), output '[PROCEDURE_FINISH]' at the very end.\n\n"
            f"AVAILABLE IMAGES:\n{image_tags}"
        )
