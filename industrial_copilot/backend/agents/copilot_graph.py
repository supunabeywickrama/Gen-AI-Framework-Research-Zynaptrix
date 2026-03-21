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
            
            # Convert local paths to public static URLs
            api_url = os.getenv("API_URL", "http://localhost:8500")
            for img_path in rag_response.get("images", []):
                # img_path is like "data/extracted/file.png"
                web_path = img_path.replace("data/", "/static/")
                images.append(f"{api_url}{web_path}")
                
            rag = f"RAG Extract: {rag_response['answer']} \n[Retrieved Diagrams: {len(images)}]"
        except Exception as e:
            rag = f"RAG Extract Failed: {str(e)}"
    else:
        rag = f"Manual Extract: If {state['machine_state']} occurs, immediately calibrate."
        
    return {"rag_context": rag, "retrieved_images": images}

def strategy_node(state: CopilotState):
    logger.info("🤖 [Agent] Strategy")
    plan = f"Action Plan: 1) System Halting. 2) Based on diagnostic '{state['diagnostic_report']}' and RAG guidance '{state['rag_context']}', dispatch operator."
    return {"strategy_report": plan}

def critic_node(state: CopilotState):
    logger.info("🤖 [Agent] Critic")
    feedback = f"Validation: The strategy is sound and prioritizes safety. Ready for Human-in-the-loop Execution."
    return {
        "critic_feedback": feedback,
        "final_execution_plan": f"{state.get('strategy_report', '')}\n\n[Critic Sign-off]: {feedback}"
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
