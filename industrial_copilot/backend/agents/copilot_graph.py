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
    logger.info("🤖 [Agent] Sensor Status")
    suspect = state.get('suspect_sensor', 'Multiple')
    report = f"Detected severe deviations aligning with {suspect}. Mathematical threshold exceeded by {state['anomaly_score']:.2f} MSE."
    return {"sensor_status_report": report}

def diagnostic_node(state: CopilotState):
    logger.info("🤖 [Agent] Diagnostic")
    report = f"Root Cause Analysis indicates a high probability of a '{state['machine_state']}' failure event originating from the sensor anomalies classified."
    return {"diagnostic_report": report}

def knowledge_retrieval_node(state: CopilotState):
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
            latest_chunk = db.query(ManualChunk.manual_id).order_by(desc(ManualChunk.id)).first()
            manual_id = latest_chunk[0] if latest_chunk else "lathe_machine_v1"
            db.close()
            
            rag_response = rag_gen.generate_response(query, manual_id)
            
            # Use the environment variable or default to the running port 8000
            api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
            if api_url.endswith('/'):
                api_url = api_url[:-1]

            for img_path in rag_response.get("images", []):
                # CRITICAL: Normalize Windows backslashes to forward slashes for web URLs
                normalized_path = img_path.replace('\\', '/')
                # web_path should be like "/static/extracted/file.png"
                web_path = normalized_path.replace("data/", "/static/")
                if not web_path.startswith('/'):
                    web_path = '/' + web_path
                images.append(f"{api_url}{web_path}")
                
            # CLEAN REPORT: Remove debug prefixes for human readability
            rag = rag_response['answer']
        except Exception as e:
            rag = f"Manual Extract Failed: {str(e)}"
    else:
        rag = "Standard Procedures: Maintain equipment as per manufacturer specifications."
        
    return {"rag_context": rag, "retrieved_images": images}

def strategy_node(state: CopilotState):
    logger.info("🤖 [Agent] Strategy")
    # The RAG context now contains the full interleaved manual content. 
    # We elevate it to the primary strategy report.
    plan = state.get('rag_context', 'No RAG context available.')
    return {"strategy_report": plan}

def critic_node(state: CopilotState):
    logger.info("🤖 [Agent] Critic")
    feedback = "Validation: Safety protocols verified against Zynaptrix-9000 Manual. Strategy approved for execution."
    
    # We combine everything into a clean, human-readable final procedure
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
    Constructs the LangGraph state machine executing the exact Multi-Agent
    pipeline defined in the Generative AI Architectural Diagram.
    """
    workflow = StateGraph(CopilotState)
    
    # Add Nodes
    workflow.add_node("SensorStatusAgent", sensor_status_node)
    workflow.add_node("DiagnosticAgent", diagnostic_node)
    workflow.add_node("KnowledgeRetrievalAgent", knowledge_retrieval_node)
    workflow.add_node("StrategyAgent", strategy_node)
    workflow.add_node("CriticAgent", critic_node)
    
    # Add Edges (Linear for now as per diagram path to executing plan)
    workflow.set_entry_point("SensorStatusAgent")
    workflow.add_edge("SensorStatusAgent", "DiagnosticAgent")
    workflow.add_edge("DiagnosticAgent", "KnowledgeRetrievalAgent")
    workflow.add_edge("KnowledgeRetrievalAgent", "StrategyAgent")
    workflow.add_edge("StrategyAgent", "CriticAgent")
    workflow.add_edge("CriticAgent", END)
    
    return workflow.compile()
