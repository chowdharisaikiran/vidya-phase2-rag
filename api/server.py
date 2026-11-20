
import os
import sys
import traceback
import base64
import json

from pathlib import Path
from typing import Optional
from uuid import uuid4
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials,
)
from pydantic import BaseModel, Field
from supabase import create_client


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env", override=True)

sys.path.insert(0, str(PROJECT_ROOT))

from rag import ask_vidya, create_vector_store


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY"
)

MAX_HISTORY_MESSAGES = 20

security = HTTPBearer(auto_error=False)


# --------------------------------------------------
# 2. FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Vidya RAG API",
    description="Vidya AI Learning Assistant - RAG API",
    version="3.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 3. SUPABASE CLIENTS
# --------------------------------------------------

def get_supabase_clients():

    if not all([
        SUPABASE_URL,
        SUPABASE_ANON_KEY,
        SUPABASE_SERVICE_ROLE_KEY,
    ]):
        raise HTTPException(
            status_code=503,
            detail="Supabase configuration is incomplete.",
        )

    auth_client = create_client(
        SUPABASE_URL,
        SUPABASE_ANON_KEY,
    )

    database_client = create_client(
        SUPABASE_URL,
        SUPABASE_SERVICE_ROLE_KEY,
    )

    # Diagnostic: inspect actual database client JWT role
    try:

        auth_header = (
            database_client.postgrest.session.headers.get(
                "Authorization",
                ""
            )
        )

        actual_token = auth_header.replace(
            "Bearer ",
            "",
            1
        )

        token_parts = actual_token.split(".")

        if len(token_parts) == 3:

            payload = token_parts[1]

            payload += "=" * (
                -len(payload) % 4
            )

            actual_claims = json.loads(
                base64.urlsafe_b64decode(payload)
            )

            print(
                "\n========== DATABASE CLIENT CHECK =========="
            )

            print(
                "ACTUAL DATABASE CLIENT ROLE:",
                actual_claims.get("role")
            )

            print(
                "DATABASE CLIENT PROJECT:",
                actual_claims.get("ref", "Not available")
            )

            print(
                "EXPECTED PROJECT:",
                SUPABASE_URL.split("//")[-1].split(".")[0]
            )

            print(
                "============================================\n"
            )

        else:

            print(
                "DATABASE CLIENT TOKEN FORMAT:",
                "Not a JWT"
            )

    except Exception as error:

        print(
            "DATABASE CLIENT TOKEN INSPECTION ERROR:",
            type(error).__name__,
            str(error)
        )

    return auth_client, database_client


# --------------------------------------------------
# 4. USER AUTHENTICATION
# --------------------------------------------------

def get_authenticated_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        security
    ),
):

    if not credentials:

        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
        )

    auth_client, database_client = get_supabase_clients()

    try:

        response = auth_client.auth.get_user(
            credentials.credentials
        )

        if not response or not response.user:

            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token.",
            )

        return {
            "user_id": str(response.user.id),
            "database": database_client,
        }

    except HTTPException:
        raise

    except Exception as error:

        print("\n========== AUTHENTICATION ERROR ==========")

        print(
            f"Error Type: {type(error).__name__}"
        )

        print(
            f"Error Message: {error}"
        )

        traceback.print_exc()

        print("==========================================\n")

        raise HTTPException(
            status_code=401,
            detail="Authentication failed.",
        )


# --------------------------------------------------
# 5. REQUEST MODEL
# --------------------------------------------------

class QuestionRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        max_length=10000,
    )

    session_id: Optional[str] = Field(
        default=None,
        max_length=200,
    )


# --------------------------------------------------
# 6. HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "application": "Vidya RAG API",
        "status": "running",
        "version": "3.0.1",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "application": "Vidya",
    }


# --------------------------------------------------
# 7. CHAT SESSION MANAGEMENT
# --------------------------------------------------

def get_or_create_session(
    database,
    user_id: str,
    session_id: Optional[str],
):

    if session_id:

        response = (
            database.table("chat_sessions")
            .select("session_id,user_id")
            .eq("session_id", session_id)
            .execute()
        )

        if response.data:

            existing = response.data[0]

            if str(existing["user_id"]) != user_id:

                raise HTTPException(
                    status_code=403,
                    detail="You cannot access this chat session.",
                )

            return session_id

    new_session_id = session_id or str(uuid4())

    database.table("chat_sessions").insert({
        "user_id": user_id,
        "session_id": new_session_id,
        "title": "New conversation",
    }).execute()

    return new_session_id


# --------------------------------------------------
# 8. LOAD CHAT HISTORY
# --------------------------------------------------

def load_chat_history(
    database,
    session_id: str,
):

    response = (
        database.table("chat_messages")
        .select("role,content")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .limit(MAX_HISTORY_MESSAGES)
        .execute()
    )

    messages = response.data or []

    messages.reverse()

    return messages


# --------------------------------------------------
# 9. SAVE CHAT MESSAGE
# --------------------------------------------------

def save_message(
    database,
    session_id: str,
    role: str,
    content: str,
):

    database.table("chat_messages").insert({
        "session_id": session_id,
        "role": role,
        "content": content,
    }).execute()


# --------------------------------------------------
# 10. DOCUMENT INGESTION
# --------------------------------------------------

@app.post("/ingest")
def ingest_documents(
    auth=Depends(get_authenticated_user),
):

    try:

        result = create_vector_store()

        return {
            "status": "success",
            "message": "Knowledge base ingestion completed.",
            "result": str(result),
        }

    except Exception as error:

        print("\n========== VIDYA INGESTION ERROR ==========")

        print(
            f"Error Type: {type(error).__name__}"
        )

        print(
            f"Error Message: {error}"
        )

        traceback.print_exc()

        print("============================================\n")

        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed.",
        )


# --------------------------------------------------
# 11. SOURCE EXTRACTION
# --------------------------------------------------

def extract_sources(answer: str):

    sources = []

    source_markers = [
        "**Knowledge Base Sources:**",
        "**Sources:**",
        "**Source:**",
    ]

    for marker in source_markers:

        if marker not in answer:
            continue

        source_text = answer.split(
            marker,
            1,
        )[1].strip()

        for line in source_text.splitlines():

            line = line.strip()

            if line.startswith("-"):

                source = line.lstrip("-").strip()

                if source and source not in sources:
                    sources.append(source)

        break

    return sources


# --------------------------------------------------
# 12. MAIN RAG CHAT ENDPOINT
# --------------------------------------------------

@app.post("/ask")
def ask_question(
    request: QuestionRequest,
    auth=Depends(get_authenticated_user),
):

    try:

        print("\n========== VIDYA ASK REQUEST ==========")

        question = request.question.strip()

        if not question:

            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty.",
            )

        user_id = auth["user_id"]
        database = auth["database"]

        print("Step 1: Authentication successful.")

        requested_session_id = (
            request.session_id.strip()
            if request.session_id
            else None
        )

        # Create or retrieve session
        session_id = get_or_create_session(
            database,
            user_id,
            requested_session_id,
        )

        print("Step 2: Chat session ready.")

        # Load conversation history
        chat_history = load_chat_history(
            database,
            session_id,
        )

        print(
            f"Step 3: Loaded {len(chat_history)} history messages."
        )

        # Generate RAG answer
        print("Step 4: Starting RAG processing.")

        answer = ask_vidya(
            question,
            chat_history=chat_history,
        )

        if not answer:

            raise ValueError(
                "RAG pipeline returned an empty answer."
            )

        print("Step 5: RAG answer generated.")

        # Extract sources from RAG answer
        sources = extract_sources(answer)

        source_type = (
            "knowledge_base"
            if sources
            else "general_ai"
        )

        print("Extracted sources:", sources)
        print("Source type:", source_type)

        # Save user message
        save_message(
            database,
            session_id,
            "user",
            question,
        )

        print("Step 6: User message saved.")

        # Save assistant message
        save_message(
            database,
            session_id,
            "assistant",
            answer,
        )

        print("Step 7: Assistant message saved.")

        # Update session timestamp
        database.table("chat_sessions").update({
            "updated_at": datetime.now(
                timezone.utc
            ).isoformat()
        }).eq(
            "session_id",
            session_id,
        ).execute()

        print("Step 8: Session timestamp updated.")

        print("========== VIDYA ASK SUCCESS ==========\n")

        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "sources": sources,
            "source_type": source_type,
            "session_id": session_id,
            "history_messages": len(chat_history) + 2,
        }

    except HTTPException:
        raise

    except Exception as error:

        print("\n========== VIDYA ASK ERROR ==========")

        print(
            f"Error Type: {type(error).__name__}"
        )

        print(
            f"Error Message: {error}"
        )

        traceback.print_exc()

        print("=====================================\n")

        raise HTTPException(
            status_code=500,
            detail=(
                "Vidya could not process this question. "
                "Please try again."
            ),
        )
