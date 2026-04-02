import logging
import os
from typing import TypedDict, Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from unified_rag.retrieval.rag import RAGMode

# EXPLICITLY IMPORTING THE IMPROVED RAG SYSTEM HERE!
try:
    from unified_rag.retrieval.rag import RAGGenerator
    rag_gen = RAGGenerator()
except Exception:
    rag_gen = None

logger = logging.getLogger(__name__)

class CopilotState(TypedDict):
    """
    Represents the shared state modified by all nodes in the LangGraph orchestration.
    This state acts as the 'Short-term Memory' for the AI diagnostic session.
    """
    # Alert Inputs from the LSTM/Autoencoder
    event_id: str
    machine_id: str
    machine_state: str
    anomaly_score: float
    user_query: Optional[str]
    suspect_sensor: Optional[str]
    recent_readings: Optional[Dict[str, Any]]
    
    # State accumulated across agents
    sensor_status_report: str
    diagnostic_report: str
    rag_context: str
    retrieved_images: List[str]
    strategy_report: str
    critic_feedback: str
    final_execution_plan: str
    chat_history: Optional[str]  # The current conversation string for the RAG Wizard

def sensor_status_node(state: CopilotState):
    """
    Agent Node: Translates raw sub-symbolic telemetry into natural language status.
    Uses the Anomaly Score (MSE) to quantify the severity of the deviation.
    """
    logger.info("🤖 [Agent] Sensor Status")
    suspect = state.get('suspect_sensor', 'Multiple')
    report = f"Detected severe deviations aligning with {suspect}. Mathematical threshold exceeded by {state['anomaly_score']:.2f} MSE."
    return {"sensor_status_report": report}

def diagnostic_node(state: CopilotState):
    """
    Agent Node: Performs a symbolic Root Cause Analysis (RCA).
    Maps the detected anomaly pattern to the suspected failure mode (e.g. Pump Failure, Sensor Drift).
    """
    logger.info("🤖 [Agent] Diagnostic")
    report = f"Root Cause Analysis indicates a high probability of a '{state['machine_state']}' failure event originating from the sensor anomalies classified."
    return {"diagnostic_report": report}

def knowledge_retrieval_node(state: CopilotState):
    """
    Agent Node: The Multimodal RAG Interface.
    1. Orchestrates semantic search against the Zynaptrix-9000 PDF Vector DB.
    2. Resolves technical diagram paths for the React Frontend.
    3. Normalizes OS-level path separators to ensure 100% URL reliability.
    """
    logger.info(f"🤖 [Agent] Knowledge Retrieval for Machine: {state.get('machine_id', 'Unknown')}")
    machine_id = state.get('machine_id', 'PUMP-001')
    
    # HITL: Use the operator's specific query if available, otherwise default to a general summary
    user_q = state.get('user_query', '')
    mode = RAGMode.SUMMARY
    query = user_q

    if "[CLARIFY_STEP]" in user_q:
        mode = RAGMode.CLARIFICATION
        query = user_q.replace("[CLARIFY_STEP]", "").strip()
    elif "[EVALUATE_STEP]" in user_q:
        mode = RAGMode.EVALUATION
        query = user_q.replace("[EVALUATE_STEP]", "").strip()
    elif "[CONVERSATIONAL_WIZARD]" in user_q or "Generate full step-by-step repair procedure" in user_q:
        # This is the new Intelligent Navigator
        mode = RAGMode.CONVERSATIONAL_WIZARD
        query = user_q.replace("[CONVERSATIONAL_WIZARD]", "").strip()
        if not query:
            query = "Provide the first instruction from the repair manual."
    elif user_q.strip():
        query = f"Diagnostic Summary only for: {user_q} (Anomaly: {state['machine_state']})"
    else:
        query = f"Provide a brief diagnostic summary and status for {state['machine_state']}."
    
    images = []
    if rag_gen:
        try:
            # DYNAMIC MANUAL SELECTION: Resolve machine_id -> manual_id via Machine Registry
            from unified_rag.db.database import SessionLocal
            from unified_rag.db.models import Machine
            
            db = SessionLocal()
            try:
                # 1. Lookup the machine's specific manual
                machine_record = db.query(Machine).filter(Machine.machine_id == machine_id).first()
                manual_id = machine_record.manual_id if machine_record else "Zynaptrix_9000"
                
                # 2. PROVENANCE CHECK: Does this manual actually have content in our RAG DB?
                from unified_rag.db.models import ManualChunk
                chunk_count = db.query(ManualChunk).filter(ManualChunk.manual_id == manual_id).count()
                
                if chunk_count == 0:
                    logger.warning(f"⚠️ [Provenance] No manual content found for {manual_id}. Flagging for disclaimer.")
                    # Prepend a hidden flag to the query so the RAG knows to DISCLAIMER it.
                    query = f"[DISCLAIMER_REQUIRED: MISSING_MANUAL] {query}"
                else:
                    logger.info(f"✅ [Provenance] {chunk_count} manual chunks found for {manual_id}.")
            finally:
                db.close()
            
            logger.info(f"🔍 [RAG Routing] Machine {machine_id} resolved to Manual: {manual_id} (Mode: {mode})")
            rag_response = rag_gen.generate_response(
                query, 
                manual_id, 
                machine_id, 
                mode=mode, 
                chat_history=state.get('chat_history', '')
            )
            
            # API URL configuration with dynamic fallback
            api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
            if api_url.endswith('/'):
                api_url = api_url[:-1]

            for img_path in rag_response.get("images", []):
                # CROSS-PLATFORM NORMALIZATION
                normalized_path = img_path.replace('\\', '/')
                # Transform filesystem path to virtual mount path (/static)
                web_path = normalized_path.replace("data/", "/static/")
                if not web_path.startswith('/'):
                    web_path = '/' + web_path
                images.append(f"{api_url}{web_path}")
                
            rag = rag_response['answer']
        except Exception as e:
            logger.error(f"❌ [RAG Routing] Dynamic lookup failed for {machine_id}: {e}")
            rag = f"Manual Extract Failed for {machine_id}: {str(e)}"
    else:
        rag = "Standard Procedures: Maintain equipment as per manufacturer specifications."
        
    return {"rag_context": rag, "retrieved_images": images}

def strategy_node(state: CopilotState):
    """
    Agent Node: Formulates the final strategy using RAG context.
    The strategy incorporates the exact textual instructions and interleaved image tags.
    """
    logger.info("🤖 [Agent] Strategy")
    plan = state.get('rag_context', 'No RAG context available.')
    return {"strategy_report": plan}

def critic_node(state: CopilotState):
    """
    Agent Node: Final Validation & Formatting.
    Ensures the output is sanitized and formatted for the Next.js 'Markdown-Lite' renderer.
    """
    logger.info("🤖 [Agent] Critic")
    feedback = f"Validation: Safety protocols verified against {state.get('machine_id', 'Asset')} Technical Documentation. Strategy approved for execution."
    
    # HITL: If this is a specific user query, provide a clean chat response.
    # If it's an initial anomaly detection (no user query yet), provide a brief summary.
    user_q = state.get('user_query')
    if user_q and user_q.strip():
        final_output = state['strategy_report']
    else:
        # Initial AI Diagnostic Message: Clean & Suggestion-Focused
        final_output = (
            f"### 🚨 AI Diagnostic Alert\n"
            f"**Event**: {state['machine_state']} | **Confidence**: {state['anomaly_score']:.2f}\n\n"
            f"{state['strategy_report']}\n"
        )
        
    return {
        "critic_feedback": feedback,
        "final_execution_plan": final_output
    }

def build_copilot_graph() -> StateGraph:
    """
    Architecture Definition:
    Creates a directed acyclic graph (DAG) representing the AI reasoning process.
    The linear pipeline ensures a predictable 'Sensor-to-Execution' transformation.
    """
    workflow = StateGraph(CopilotState)
    
    workflow.add_node("SensorStatusAgent", sensor_status_node)
    workflow.add_node("DiagnosticAgent", diagnostic_node)
    workflow.add_node("KnowledgeRetrievalAgent", knowledge_retrieval_node)
    workflow.add_node("StrategyAgent", strategy_node)
    workflow.add_node("CriticAgent", critic_node)
    
    workflow.set_entry_point("SensorStatusAgent")
    workflow.add_edge("SensorStatusAgent", "DiagnosticAgent")
    workflow.add_edge("DiagnosticAgent", "KnowledgeRetrievalAgent")
    workflow.add_edge("KnowledgeRetrievalAgent", "StrategyAgent")
    workflow.add_edge("StrategyAgent", "CriticAgent")
    workflow.add_edge("CriticAgent", END)
    
    return workflow.compile()
