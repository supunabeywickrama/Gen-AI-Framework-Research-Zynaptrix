import logging
import json
import openai
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc

from unified_rag.config import settings
from unified_rag.retrieval.rag import RAGGenerator, RAGMode
from unified_rag.db.database import get_db
from unified_rag.db.models import AssistantSession, AssistantMessage

router = APIRouter()
logger = logging.getLogger(__name__)
openai.api_key = settings.openai_api_key

class AssistantQuery(BaseModel):
    query: str
    session_id: Optional[int] = None
    machine_id: Optional[str] = None # Optional context for RAG

class AssistantSessionResponse(BaseModel):
    id: int
    machine_id: Optional[str]
    title: str
    timestamp: str

class AssistantMessageView(BaseModel):
    role: str
    content: str
    type: str
    step_data: Optional[Dict]
    images: List[str]
    timestamp: str

# SYSTEM KNOWLEDGE (Internal Documentation for the Assistant)
SYSTEM_ONBOARDING_CONTEXT = """
SYSTEM: Industrial Copilot Hub — Zynaptrix Platform

MODULES:
1. DASHBOARD: Real-time sensor telemetry charts. Select a machine from the top-bar MachineSelector to filter charts.
2. INCIDENT REGISTRY: Lists active anomaly alerts detected by the AI engine. Click an alert to open the Diagnostic Copilot chat.
3. DIAGNOSTIC COPILOT CHAT: AI-guided repair wizard for anomaly incidents. Provides step-by-step repair procedures.
4. CENTRAL ASSISTANT (this bot): A general knowledge assistant. Select a machine manual from the dropdown in the chat header to query RAG manuals.
5. MANUAL INGESTION (Knowledge Ingestion Tab): Upload PDF manuals for machines.
6. SIMULATOR: Start/Stop real-time telemetry simulation per machine.

GEN AI FRAMEWORK ARCHITECTURE:
This system represents a state-of-the-art "Gen AI Framework" for industrial environments.
- Multi-Agent Orchestration: Powered by LangGraph, multiple AI agents (Sensor Analysis, Diagnostic, Strategy, Critic, RAG) collaborate asynchronously to analyze anomalies and recommend procedures.
- Retrieval-Augmented Generation (RAG): Uses pgvector and OpenAI embeddings to semantically search technical manuals isolated per machine.
- Interactive Wizard Engine: Automatically converts dense technical manuals into sequential, conversational step-by-step UI actions.
- Adaptive Learning (Interaction Memory): As users confirm or reject maintenance steps, the framework stores the successful workflows back into the vector DB for future automated resolution.

HOW TO INGEST A MANUAL:
Step 1: Navigate to the Knowledge Ingestion tab from the left navigation.
Step 2: Enter the Manual ID — this must exactly match the manual_id you used when registering the machine in the database.
Step 3: Upload the PDF file using the file picker.
Step 4: Click the "Initial Ingestion" button to start processing. The system will chunk, embed, and store the manual in the vector database.
Step 5: After ingestion completes, go to the machine registration section and link the manual_id to the machine.

HOW TO REGISTER A MACHINE & ADD SENSORS:
Step 1: Go to the machine management section (Machine Registry).
Step 2: Enter the Machine ID (e.g. PUMP-001), Machine Name, Location, and Manual ID.
Step 3: Click "Add Sensor" to configure individual sensors. For each sensor, enter the Sensor Name, Sensor ID, and upload its manufacturer Datasheet (PDF).
Step 4: Save the machine.
What happens next? The system's Gen AI Framework automatically parses the uploaded sensor datasheets to extract operational parameters, generates synthentic telemetry data datasets, and trains a dedicated Anomaly Detection Machine Learning model strictly for that machine in the background.

HOW TO START THE SIMULATOR:
Step 1: Select a machine from the top-bar MachineSelector.
Step 2: Click the "Start" button in the top-right of the dashboard header.
Step 3: The simulator will begin streaming live telemetry data to the charts.

HOW TO USE THE ASSISTANT WITH A MACHINE MANUAL:
Step 1: Open the Central Assistant by clicking the blue chat button (bottom-right).
Step 2: Use the "RAG: [Machine] Manual" dropdown in the chat header to select a machine.
Step 3: Ask any technical question. The assistant will query that machine's specific manual and answer with expert guidance.

SECURITY NOTES:
- All API endpoints require proper session context.
- Machine manuals are isolated per machine_id in the vector database.
- Chat history is stored in the assistant_sessions and assistant_messages tables.
- Anomaly data is stored in anomaly_records table.
"""

# Structured step definitions for system guide topics
SYSTEM_GUIDE_STEPS = {
    "ingest": [
        {"id": "ingest_1", "title": "Navigate to Knowledge Ingestion Tab", "detail": "Go to the Ingestion page from the navigation sidebar."},
        {"id": "ingest_2", "title": "Enter the Manual ID", "detail": "Add the same manual_id you entered when registering the machine. This links the manual to the machine."},
        {"id": "ingest_3", "title": "Upload the PDF file", "detail": "Click the file picker and select the machine manual PDF from your local filesystem."},
        {"id": "ingest_4", "title": "Click the 'Initial Ingestion' button", "detail": "This starts the pipeline: the system will chunk, embed, and store the manual in the vector database (pgvector)."},
        {"id": "ingest_5", "title": "Wait for completion", "detail": "The ingestion log will show progress. Once done, the manual is searchable via RAG."},
    ],
    "register": [
        {"id": "reg_1", "title": "Open Machine Management", "detail": "Navigate to the machine registration section in the admin panel."},
        {"id": "reg_2", "title": "Enter Machine Details", "detail": "Fill in Machine ID (e.g. PUMP-001), Machine Name, Location, and the Manual ID that matches your ingested PDF."},
        {"id": "reg_3", "title": "Add Sensors & Datasheets", "detail": "Click 'Add Sensor'. Enter the Sensor ID and Name, then upload the manufacturer's PDF datasheet for that sensor."},
        {"id": "reg_4", "title": "Save & Train Model", "detail": "Click Save. The Gen AI Framework will securely parse the datasheets, generate a synthetic dataset, and automatically train an anomaly detection ML model for this machine."},
    ],
    "simulator": [
        {"id": "sim_1", "title": "Select a Machine", "detail": "Use the top-bar MachineSelector dropdown to choose which machine to simulate."},
        {"id": "sim_2", "title": "Click the Start Button", "detail": "Click the green 'Start' button in the dashboard header. The simulator begins streaming live telemetry."},
        {"id": "sim_3", "title": "Monitor Telemetry Charts", "detail": "Live sensor data will populate the charts in real-time. An anomaly alert will fire automatically if readings exceed thresholds."},
    ],
    "assistant": [
        {"id": "ast_1", "title": "Open the Central Assistant", "detail": "Click the blue chat bubble button in the bottom-right corner of the dashboard."},
        {"id": "ast_2", "title": "Select a Machine Manual (optional)", "detail": "Use the 'RAG: [Machine] Manual' dropdown in the chat header to target a specific machine's documentation."},
        {"id": "ast_3", "title": "Ask Your Question", "detail": "Type any technical question. If a machine is selected, the answer comes from that machine's manual via RAG. Otherwise it uses general AI knowledge."},
    ],
    "framework": [
        {"id": "fw_1", "title": "Multi-Agent Orchestration Engine", "detail": "Under the hood, LangGraph coordinates multiple AI agents (Sensor Analysis, Strategy, RAG, Critic) to process anomalies entirely autonomously."},
        {"id": "fw_2", "title": "Adaptive Interaction Intelligence", "detail": "As you interact with step-cards (e.g., clicking 'I have done' or 'I am stuck'), the system learns successful resolution paths, saving them back into pgvector via the Interaction Memory."},
        {"id": "fw_3", "title": "Multi-Modal RAG Injection", "detail": "Technical queries dynamically extract relevant text and images from PDF manuals, synthesizing them into Conversational Wizard cards directly in the chat interface."},
    ]
}

def detect_guide_topic(query: str) -> str | None:
    """Detect if query is asking about system usage and return the topic key."""
    q = query.lower()
    if any(x in q for x in ["ingest", "upload", "manual", "pdf", "add manual", "import"]):
        return "ingest"
    if any(x in q for x in ["register machine", "add machine", "create machine", "add sensor", "add sensors", "datasheet", "train model"]):
        return "register"
    if any(x in q for x in ["simulator", "start simulation", "simulate", "telemetry stream"]):
        return "simulator"
    if any(x in q for x in ["how to use assistant", "use this bot", "rag manual", "assistant help"]):
        return "assistant"
    if any(x in q for x in ["framework", "architecture", "system works", "gen ai", "how this works", "what is this"]):
        return "framework"
    return None

def generate_session_title(query: str) -> str:
    """ Generate a 3-4 word title for a new session based on the first query. """
    try:
        res = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Generate a very short title (max 4 words) for a chat that starts with this question: '{query}'. Provide ONLY the title."}],
            max_tokens=20
        )
        return res.choices[0].message.content.strip().strip('"')
    except:
        return "New Assistant Inquiry"

def perform_web_search(query: str) -> str:
    # ... (existing simulated search) ...
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

# --- New Session Management Endpoints ---

@router.get("/api/assistant/sessions", response_model=List[AssistantSessionResponse])
async def get_assistant_sessions(db: Session = Depends(get_db)):
    sessions = db.query(AssistantSession).order_by(desc(AssistantSession.updated_at)).all()
    return [{
        "id": s.id,
        "machine_id": s.machine_id,
        "title": s.title,
        "timestamp": s.updated_at
    } for s in sessions]

@router.delete("/api/assistant/sessions/{session_id}")
async def delete_assistant_session(session_id: int, db: Session = Depends(get_db)):
    db.query(AssistantMessage).filter(AssistantMessage.session_id == session_id).delete()
    db.query(AssistantSession).filter(AssistantSession.id == session_id).delete()
    db.commit()
    return {"status": "success"}

@router.get("/api/assistant/sessions/{session_id}/history", response_model=List[AssistantMessageView])
async def get_session_history(session_id: int, db: Session = Depends(get_db)):
    messages = db.query(AssistantMessage).filter(AssistantMessage.session_id == session_id).order_by(AssistantMessage.timestamp).all()
    return [{
        "role": m.role,
        "content": m.content,
        "type": m.type,
        "step_data": json.loads(m.step_data) if m.step_data else None,
        "images": json.loads(m.images) if m.images else [],
        "timestamp": m.timestamp
    } for m in messages]

@router.get("/api/assistant/sessions/{session_id}/export")
async def export_session(session_id: int, db: Session = Depends(get_db)):
    """
    Export session data formatted for PDF generation.
    Includes session metadata and full message history with images.
    """
    try:
        # Get session info
        session = db.query(AssistantSession).filter(AssistantSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get all messages
        messages = db.query(AssistantMessage).filter(
            AssistantMessage.session_id == session_id
        ).order_by(AssistantMessage.timestamp).all()
        
        # Format messages for PDF
        formatted_messages = []
        for m in messages:
            formatted_messages.append({
                "role": m.role,
                "content": m.content,
                "images": json.loads(m.images) if m.images else [],
                "timestamp": m.timestamp,
                "context_source": None  # Can be added later if needed
            })
        
        return {
            "session_id": session.id,
            "session_title": session.title,
            "machine_context": session.machine_id,
            "created_at": session.created_at,
            "messages": formatted_messages
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export session")

@router.get("/api/assistant/sessions/{session_id}/report")
async def generate_report(session_id: int, db: Session = Depends(get_db)):
    """
    Generate structured diagnostic report from session.
    Extracts problem, diagnosis, solution steps, and images using AI.
    """
    try:
        # Get session info
        session = db.query(AssistantSession).filter(AssistantSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get all messages
        messages = db.query(AssistantMessage).filter(
            AssistantMessage.session_id == session_id
        ).order_by(AssistantMessage.timestamp).all()
        
        if not messages:
            raise HTTPException(status_code=400, detail="No messages in session")
        
        # Compile conversation
        conversation = []
        all_images = []
        
        for m in messages:
            conversation.append(f"{m.role.upper()}: {m.content}")
            if m.images:
                try:
                    imgs = json.loads(m.images)
                    all_images.extend(imgs)
                except:
                    pass
        
        conversation_text = "\n\n".join(conversation)
        logger.info(f"Report generation - Session {session_id}: {len(messages)} messages, {len(all_images)} images")
        logger.info(f"Conversation text length: {len(conversation_text)} chars")
        
        # Use OpenAI to extract structured information
        system_prompt = """You are a technical documentation expert. Extract key information from this diagnostic conversation and structure it as a professional maintenance report.

Analyze the conversation carefully and extract:
1. PROBLEM: What issue or problem was reported? Summarize in 2-3 clear sentences.
2. DIAGNOSIS: What was identified as the root cause or analysis? Summarize in 2-4 sentences.
3. SOLUTION_STEPS: Extract ALL repair steps, instructions, or procedures mentioned. Each step should be clear and actionable.

IMPORTANT: Extract the ACTUAL content from the conversation. Do not use placeholder text.

You MUST respond with ONLY valid JSON in this exact format (no markdown, no code blocks):
{"problem": "actual problem description from conversation", "diagnosis": "actual diagnosis from conversation", "solution_steps": ["step 1 from conversation", "step 2 from conversation"]}"""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract report data from this conversation:\n\n{conversation_text}"}
            ],
            temperature=0.2,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        ai_response = response.choices[0].message.content
        logger.info(f"AI Response for report: {ai_response[:500]}...")
        
        # Parse AI response
        try:
            extracted = json.loads(ai_response)
        except Exception as parse_err:
            logger.error(f"Failed to parse AI response: {parse_err}")
            # If AI response parsing fails, create report from raw conversation
            extracted = {
                "problem": conversation[0] if conversation else "No problem described",
                "diagnosis": "Review conversation for detailed analysis",
                "solution_steps": [msg for msg in conversation if msg.startswith("ASSISTANT:")][:5]
            }
        
        # Validate extracted content is not empty
        if not extracted.get("problem") or extracted.get("problem") == "":
            extracted["problem"] = conversation[0] if conversation else "No problem description available"
        if not extracted.get("diagnosis") or extracted.get("diagnosis") == "":
            extracted["diagnosis"] = "Refer to conversation for diagnosis details"
        if not extracted.get("solution_steps") or len(extracted.get("solution_steps", [])) == 0:
            extracted["solution_steps"] = ["Review conversation for detailed steps"]
        
        # Format images with captions
        formatted_images = []
        for idx, img_url in enumerate(all_images):
            formatted_images.append({
                "url": img_url,
                "caption": f"Reference Diagram {idx + 1}"
            })
        
        logger.info(f"Report generated - Problem: {extracted.get('problem', '')[:100]}...")
        
        return {
            "sessionId": session.id,
            "machineId": session.machine_id,
            "problemDescription": extracted.get("problem", "No problem description available"),
            "diagnosis": extracted.get("diagnosis", "No diagnosis available"),
            "solutionSteps": extracted.get("solution_steps", []),
            "images": formatted_images,
            "timestamp": session.created_at,
            "operatorNotes": None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate report for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@router.post("/api/copilot/assistant")
async def system_assistant(req: AssistantQuery, db: Session = Depends(get_db)):
    """
    The Stateful Assistant Endpoint.
    """
    try:
        # 🧵 PHASE 0: Session Management
        active_session_id = req.session_id
        active_machine_id = req.machine_id

        if active_session_id:
            session = db.query(AssistantSession).filter(AssistantSession.id == active_session_id).first()
            if session:
                # Update session machine context if explicitly provided
                if req.machine_id:
                    session.machine_id = req.machine_id
                    active_machine_id = req.machine_id
                elif not active_machine_id:
                    active_machine_id = session.machine_id
        else:
            # Create new session
            new_title = generate_session_title(req.query)
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_session = AssistantSession(
                machine_id=active_machine_id,
                title=new_title,
                created_at=now_str,
                updated_at=now_str
            )
            db.add(new_session)
            db.commit()
            db.refresh(new_session)
            active_session_id = new_session.id
        
        # Save User Message
        user_msg = AssistantMessage(
            session_id=active_session_id,
            role="user",
            content=req.query,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        db.add(user_msg)
        db.commit()

        # 🧵 PHASE 0.5: Context Fetching (Conversation Memory)
        # Fetch last 10 messages for context, excluding the one we just added to keep logic clean
        messages_db = db.query(AssistantMessage).filter(
            AssistantMessage.session_id == active_session_id
        ).order_by(AssistantMessage.timestamp.desc()).offset(1).limit(10).all()
        
        chat_context = []
        # Reverse to get chronological order
        for m in reversed(messages_db):
            role = "assistant" if m.role == "agent" else m.role
            chat_context.append({"role": role, "content": m.content})

        # 🧠 PHASE 1: Intent Classification (enhanced)
        # First check if it's a system guide topic (highest priority)
        guide_topic = detect_guide_topic(req.query)
        
        if not guide_topic:
            intent_prompt = (
                "Classify this user query into ONE of these intents:\n"
                "- ONBOARDING: Questions about how this Industrial Copilot system works, its features, navigation.\n"
                "- RAG: Technical questions about machine maintenance, specs, repairs, faults — needs manual lookup.\n"
                "- SEARCH: General internet knowledge, industry news, standards.\n"
                "- CHAT: Simple greetings or off-topic conversational fillers.\n\n"
                f"Query: '{req.query}'\n"
                "Respond with ONLY one word: ONBOARDING, RAG, SEARCH, or CHAT."
            )
            intent_res = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a specialized router."}] + chat_context + [{"role": "user", "content": intent_prompt}],
                max_tokens=10, temperature=0
            )
            intent = intent_res.choices[0].message.content.strip().upper()
            # Force RAG if machine is selected and query is technical
            is_technical = any(x in req.query.lower() for x in ["repair", "fix", "step", "how to", "guide", "procedure", "fault", "error", "maintenance", "spec", "troubleshoot", "explain"])
            if active_machine_id and is_technical and intent in ("CHAT", "ONBOARDING"):
                intent = "RAG"
        else:
            intent = "GUIDE"

        # 🚀 PHASE 2: Execute by Intent
        context = "General AI Knowledge"
        msg_type = "text"
        step_data = None
        images = []
        final_answer = ""

        machine_context_prefix = ""
        if active_machine_id:
            machine_context_prefix = (
                f"CRITICAL CONTEXT: The user has selected the '{active_machine_id}' machine manual. "
                f"Always answer in the context of {active_machine_id}. "
                f"Do NOT ask which machine the user means — it is {active_machine_id}.\n\n"
            )

        if intent == "GUIDE":
            # Build a clean numbered markdown answer from the structured steps
            steps = SYSTEM_GUIDE_STEPS.get(guide_topic, [])
            lines = []
            for i, s in enumerate(steps, 1):
                lines.append(f"**Step {i}: {s['title']}**\n\n{s['detail']}")
            final_answer = "\n\n---\n\n".join(lines) if lines else "No steps defined for this topic."
            context = "System Guide"

        elif intent == "ONBOARDING":
            system_prompt = (
                f"You are the Industrial Copilot System Assistant for the Zynaptrix platform. "
                f"Answer based on this system documentation:\n\n{SYSTEM_ONBOARDING_CONTEXT}\n\n"
                f"Be concise, clear, and practical. Use numbered steps where relevant. Format your response in plain markdown."
            )
            messages = [{"role": "system", "content": f"{machine_context_prefix}{system_prompt}"}] + chat_context + [{"role": "user", "content": req.query}]
            final_res = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000
            )
            final_answer = final_res.choices[0].message.content.strip()
            context = "System Documentation"

        elif intent == "RAG" and active_machine_id:
            rag_gen = RAGGenerator()
            is_procedural = any(x in req.query.lower() for x in ["how to", "step", "procedure", "guide", "repair", "troubleshoot", "fix", "replace", "maintain"])
            mode = RAGMode.CONVERSATIONAL_WIZARD if is_procedural else RAGMode.SUMMARY
            rag_res = rag_gen.generate_response(req.query, "Technical_Manual", active_machine_id, mode=mode)
            rag_answer = rag_res.get('answer', '')
            images = rag_res.get('images', [])  # Corrected key from 'retrieved_images'
            context = f"Manual Lookup ({active_machine_id})"

            image_tags = "\n".join([f"  - [IMAGE_{i}] for document reference {i}" for i in range(len(images))]) if images else "No diagrams available."

            system_prompt = (
                f"{machine_context_prefix}"
                f"You are an expert maintenance engineer for {active_machine_id}. "
                f"Based on the following extracted manual content, provide a highly structured, clear, numbered step-by-step answer.\n\n"
                f"MANUAL CONTENT:\n{rag_answer}\n\n"
                f"AVAILABLE DIAGRAMS:\n{image_tags}\n\n"
                f"RULES:\n"
                f"1. Use numbered steps for procedures.\n"
                f"2. You MUST interleave [IMAGE_N] tags (e.g. [IMAGE_0]) directly into your steps where the visual reference is most helpful.\n"
                f"3. Be specific and technical. Plain markdown only."
            )
            messages = [{"role": "system", "content": system_prompt}] + chat_context + [{"role": "user", "content": req.query}]
            final_res = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1500
            )
            final_answer = final_res.choices[0].message.content.strip()
            final_answer = final_answer.replace("[CONVERSATIONAL_WIZARD]", "").replace("[PROCEDURE_FINISH]", "✅ Complete").strip()

        elif intent == "SEARCH":
            search_data = perform_web_search(req.query)
            system_prompt = f"Synthesize these industrial search results into a clear answer:\n{search_data}"
            messages = [{"role": "system", "content": f"{machine_context_prefix}{system_prompt}"}] + chat_context + [{"role": "user", "content": req.query}]
            final_res = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=800
            )
            final_answer = final_res.choices[0].message.content.strip()
            context = "External Knowledge"

        else:  # CHAT
            system_prompt = (
                "You are the Industrial Copilot System Assistant for the Zynaptrix platform. "
                "You help operators and engineers navigate and use the system. "
                "If no machine is selected, encourage the user to select one from the dropdown for specific manual queries."
            )
            messages = [{"role": "system", "content": f"{machine_context_prefix}{system_prompt}"}] + chat_context + [{"role": "user", "content": req.query}]
            final_res = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=400
            )
            final_answer = final_res.choices[0].message.content.strip()

        # 💾 PHASE 3: Persist and Return
        agent_msg = AssistantMessage(
            session_id=active_session_id,
            role="agent",
            content=final_answer,
            type="text",
            step_data=None,
            images=json.dumps(images) if images else None,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        db.add(agent_msg)

        # Update session timestamp
        session_obj = db.query(AssistantSession).filter(AssistantSession.id == active_session_id).first()
        if session_obj:
            session_obj.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        db.commit()

        return {
            "role": "agent",
            "content": final_answer,
            "type": "text",
            "step_data": None,
            "images": images,
            "session_id": active_session_id,
            "context_source": context,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        logger.error(f"Assistant Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
