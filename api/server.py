
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag import ask_vidya, create_vector_store


# =====================================
# APP CONFIGURATION
# =====================================

app = FastAPI(
    title="Vidya RAG API",
    description="AI-powered conversational learning assistant",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================
# SESSION MEMORY
# =====================================

conversation_sessions = {}

MAX_HISTORY_MESSAGES = 20


# =====================================
# REQUEST MODEL
# =====================================

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = None


# =====================================
# HOME
# =====================================

@app.get("/")
def home():
    return {
        "message": "Vidya RAG API is running!",
        "status": "success",
        "version": "2.0.0"
    }


# =====================================
# HEALTH CHECK
# =====================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Vidya RAG API"
    }


# =====================================
# DOCUMENT INGESTION
# =====================================

@app.post("/ingest")
def ingest_documents():
    try:
        chunks = create_vector_store()

        return {
            "message": "Documents indexed successfully",
            "chunks_indexed": chunks,
            "status": "success"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =====================================
# CONVERSATIONAL RAG
# =====================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    try:
        question = request.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        session_id = (
            request.session_id.strip()
            if request.session_id
            else None
        )

        if session_id == "":
            raise HTTPException(
                status_code=400,
                detail="Session ID cannot be empty"
            )

        # Get history for this session.
        # Requests without a session ID remain stateless.
        chat_history = (
            conversation_sessions.get(session_id, []).copy()
            if session_id
            else []
        )

        # Generate conversational RAG answer.
        answer = ask_vidya(
            question,
            chat_history=chat_history
        )

        # Extract source information.
        source_marker = "**Knowledge Base Sources:**"

        if source_marker in answer:

            answer_text, source_text = answer.split(
                source_marker,
                1
            )

            sources = [
                item.strip()
                for item in source_text.split(",")
                if item.strip()
            ]

        else:
            answer_text = answer
            sources = []

        answer_text = answer_text.strip()

        # Save conversation for this session.
        if session_id:

            chat_history.append({
                "role": "user",
                "content": question
            })

            chat_history.append({
                "role": "assistant",
                "content": answer_text
            })

            conversation_sessions[session_id] = (
                chat_history[-MAX_HISTORY_MESSAGES:]
            )

        return {
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "source_type": (
                "knowledge_base" if sources else "general_ai"
            ),
            "session_id": session_id,
            "history_messages": len(
                conversation_sessions.get(session_id, [])
            ) if session_id else 0,
            "status": "success"
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
