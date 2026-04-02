import logging
import json
import os
import openai
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

from unified_rag.config import settings
from unified_rag.retrieval.rag import RAGGenerator, RAGMode

router = APIRouter()
logger = logging.getLogger(__name__)
openai.api_key = settings.openai_api_key

# Memory-only session for assistant if not persisted to DB
assistant_sessions: Dict[str, List[Dict[str, str]]] = {}

class AssistantQuery(BaseModel):
    query: str
    session_id: str = "default_system_assistant"
    machine_id: Optional[str] = None # Optional context for RAG

# SYSTEM KNOWLEDGE (Internal Documentation for the Assistant)
SYSTEM_ONBOARDING_CONTEXT = """
SYSTEM OVERVIEW:
The Zynaptrix-9000 Industrial Copilot is a multi-agent AI platform for predictive maintenance.

CORE CAPABILITIES:
1. ASSET REGISTRY:
   - Users can register new machines (e.g. pumps, turbines, lathes) via the 'Asset Registry' tab.
   - Requires: Machine ID, Manual ID (for RAG), and a list of sensor IDs.
   - The system can automatically 'estimate' sensor configurations from technical PDFs.

2. MANUAL INGESTION:
   - Users upload PDF technical manuals in the 'Ingestion' tab.
   - The system processes these into a Vector Database with multimodal support (text + images).
   - This knowledge is used by the AI Mentor during repairs.

3. ANOMALY DETECTION:
   - Real-time telemetry is processed using an LSTM/Autoencoder model.
   - Every machine has a 'Health Score' (100 is perfect).
   - Health scores below 85% typically trigger an 'AI Diagnostic Alert'.
   - Alerts appear in the 'Incident Registry'.

4. REPAIR WIZARD (COPILOT):
   - When an anomaly is detected, the technician clicks the alert to start a guided repair.
   - The Copilot uses RAG to provide step-by-step instructions from the manual.
   - Success is recorded in 'Interaction Memory' for future learning.
"""

def perform_web_search(query: str) -> str:
    """
    Simulated Web Search capability using OpenAI with a 'Searcher' persona.
    In a production system, this would call Serper/Google Search API.
    """
    logger.info(f"🌐 Performing simulated web search for: {query}")
    try:
        search_prompt = (
            f"You are a specialized Web Search Engine for Industrial IoT and Maintenance. "
            f"Provide a summary of the latest information regarding: '{query}'. "
            f"Include real-world trends, safety standards (like ISO/OSHA), and industry news if applicable."
        )
        res = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": search_prompt}],
            max_tokens=600
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Search service unavailable: {str(e)}"

@router.post("/api/copilot/assistant")
async def system_assistant(req: AssistantQuery):
    """
    The Independent System Assistant Endpoint.
    Routes queries between:
    - Platform Onboarding (System knowledge)
    - Manual Knowledge (RAG)
    - General Knowledge (LLM/Web Search)
    """
    try:
        # 🧠 PHASE 1: Intent Classification (Onboarding vs RAG vs Search)
        intent_prompt = (
            "Classify the user query into one of these intents:\n"
            "- ONBOARDING: Questions about how to use the app (registration, ingestion, anomalies).\n"
            "- SEARCH: Questions requiring internet data or general industry knowledge.\n"
            "- RAG: Technical questions specifically about a machine's maintenance or manuals.\n"
            "- CHAT: General greeting or casual conversation.\n\n"
            f"Query: '{req.query}'\n\n"
            "Respond with ONLY the classification (ONBOARDING, SEARCH, RAG, CHAT)."
        )
        intent_res = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": intent_prompt}],
            max_tokens=10,
            temperature=0
        )
        intent = intent_res.choices[0].message.content.strip().upper()
        logger.info(f"🤖 Assistant Intent: {intent}")

        # 🚀 PHASE 2: Execute Capabilities
        if intent == "ONBOARDING":
            system_prompt = (
                "You are the Zynaptrix System Assistant. Your job is to TEACH the user how to use this platform. "
                "Use the following internal documentation to provide a clear, pointwise guide.\n\n"
                f"SYSTEM KNOWLEDGE:\n{SYSTEM_ONBOARDING_CONTEXT}"
            )
            context = "System Documentation"
        
        elif intent == "SEARCH":
            search_data = perform_web_search(req.query)
            system_prompt = (
                "You are an Industrial Intelligence Assistant. You just performed a web search to answer the user's question. "
                "Synthesize the search results into a helpful, professional response.\n\n"
                f"SEARCH RESULTS:\n{search_data}"
            )
            context = "External Web Search"
        
        elif intent == "RAG" and req.machine_id:
            # Proxy to the existing RAG Generator
            rag_gen = RAGGenerator()
            # Fetch for the specific machine mentioned or the current context
            rag_res = rag_gen.generate_response(req.query, "Technical_Manual", req.machine_id, mode=RAGMode.SUMMARY)
            system_prompt = (
                "You are an Industrial Technical Expert. You just retrieved specific manual content for the machine. "
                "Answer the user's question clearly.\n\n"
                f"MANUAL CONTEXT:\n{rag_res['answer']}"
            )
            context = f"Manual Lookup ({req.machine_id})"
        
        else:
            system_prompt = "You are a helpful, professional Assistant for the Zynaptrix Industrial Copilot platform. Answer the user's general query."
            context = "General AI Knowledge"

        # 🛡️ PHASE 3: Generate Final Answer
        final_res = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.query}
            ],
            max_tokens=1500
        )
        answer = final_res.choices[0].message.content

        return {
            "role": "agent",
            "content": answer,
            "context_source": context,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        logger.error(f"Assistant Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
