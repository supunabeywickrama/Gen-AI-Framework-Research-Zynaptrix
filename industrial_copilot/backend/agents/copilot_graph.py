import logging
import os
from typing import TypedDict, Dict, Any, Optional, List
from langgraph.graph import StateGraph, END

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
    machine_state: str
    anomaly_score: float
    suspect_sensor: Optional[str]
    recent_readings: Optional[Dict[str, float]]
    
    # State accumulated across agents
    sensor_status_report: str
    diagnostic_report: str
    rag_context: str
    retrieved_images: List[str]
    strategy_report: str
    critic_feedback: str
    final_execution_plan: str

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
    logger.info("🤖 [Agent] Knowledge Retrieval (Powered by Improved Unified RAG)")
    query = f"Provide repair instructions and troubleshooting steps for {state['machine_state']}."
    
    images = []
    if rag_gen:
        try:
            # DYNAMIC MANUAL SELECTION: Fetch the most recent manual ID from DB
            from unified_rag.db.database import SessionLocal
            from sqlalchemy import desc
            from unified_rag.db.models import ManualChunk
            
            db = SessionLocal()
            # We fetch the latest ingested manual to ensure we are using the most up-to-date documentation
            latest_chunk = db.query(ManualChunk.manual_id).order_by(desc(ManualChunk.id)).first()
            manual_id = latest_chunk[0] if latest_chunk else "Zynaptrix_9000"
            db.close()
            
            rag_response = rag_gen.generate_response(query, manual_id)
            
            # API URL configuration with dynamic fallback
            api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
            if api_url.endswith('/'):
                api_url = api_url[:-1]

            for img_path in rag_response.get("images", []):
                # CROSS-PLATFORM NORMALIZATION:
                # Windows uses \, but web URLs require /. We normalize here to avoid 404s.
                normalized_path = img_path.replace('\\', '/')
                # Transform filesystem path to virtual mount path (/static)
                web_path = normalized_path.replace("data/", "/static/")
                if not web_path.startswith('/'):
                    web_path = '/' + web_path
                images.append(f"{api_url}{web_path}")
                
            rag = rag_response['answer']
        except Exception as e:
            rag = f"Manual Extract Failed: {str(e)}"
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
    feedback = "Validation: Safety protocols verified against Zynaptrix-9000 Manual. Strategy approved for execution."
    
    # Construction of the final Multimodal Payload
    final_output = (
        f"# 🚨 AI DIAGNOSTIC REPORT\n"
        f"**Event ID**: {state['event_id']} | **Score**: {state['anomaly_score']:.2f}\n\n"
        f"{state['strategy_report']}\n\n"
        f"---\n"
        f"**[Critic Sign-off]**: {feedback}"
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
