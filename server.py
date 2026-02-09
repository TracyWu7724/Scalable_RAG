from contextlib import asynccontextmanager
import traceback

import pathlib
from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, Depends, HTTPException, status
import asyncio
import os
import uuid

from faiss_rag import myRAG_API
from db import init_db, async_session, crud
from db.models import User
from auth import get_current_user, hash_password
from schemas import (
    AskRequest, AskResponse, UserCreate, UserResponse,
    ConversationSummary, ConversationDetail,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")

rag_cache = {}

MODEL_NAME_MAP = {
    "BAAI_bge-base-en-v1-5": "BAAI/bge-base-en-v1.5",
    "intfloat_e5-base-v2": "intfloat/e5-base-v2",
    "nomic-ai_nomic-embed-text-v1": "nomic-ai/nomic-embed-text-v1",
    "sentence-transformers_all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}


LLM_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemma-3-4b-it",
]


def load_rag(model_name: str, llm_model: str = "gemini-2.5-flash"):
    cache_key = f"{model_name}::{llm_model}"
    if cache_key in rag_cache:
        return rag_cache[cache_key]

    meta_path = f"{EMBEDDINGS_DIR}/embeddings_{model_name}_meta_2.jsonl"
    index_path = f"{EMBEDDINGS_DIR}/embeddings_{model_name}_2.index"

    embed_model_name = MODEL_NAME_MAP.get(model_name)
    if embed_model_name is None:
        raise ValueError(f"Unknown embedding model: {model_name}")

    if llm_model not in LLM_MODELS:
        raise ValueError(f"Unknown LLM model: {llm_model}")

    rag = myRAG_API(
        metadata_path=meta_path,
        faiss_path=index_path,
        gemini_model_name=llm_model,
        embed_model_name=embed_model_name,
        use_reranker=True,
        reranker_model_name="Alibaba-NLP/gte-reranker-modernbert-base",
    )

    rag_cache[cache_key] = rag
    return rag


async def timeout_wrapper(fn, timeout=120):
    try:
        return await asyncio.wait_for(fn, timeout)
    except asyncio.TimeoutError:
        return "Session timed out due to inactivity."


# ── App lifespan ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    print("Database tables created.")
    yield


app = FastAPI(title="Henkel RAG API", lifespan=lifespan)


# ── Auth endpoints ─────────────────────────────────────────────────

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(body: UserCreate):
    async with async_session() as session:
        existing = await crud.get_user_by_username(session, body.username)
        if existing:
            raise HTTPException(status_code=400, detail="Username already taken")
        hashed = hash_password(body.password)
        user = await crud.create_user(session, body.username, hashed)
        return user


# ── Ask endpoint (POST with auth + conversation persistence) ──────

@app.post("/ask", response_model=AskResponse)
async def ask_post(body: AskRequest, user: User = Depends(get_current_user)):
    try:
        rag = load_rag(body.model_name, body.llm_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load RAG model: {e}")

    try:
        async with async_session() as session:
            # Create or load conversation
            if body.conversation_id is None:
                # New conversation -- use first 60 chars of question as title
                title = body.question[:60] + ("..." if len(body.question) > 60 else "")
                conv = await crud.create_conversation(
                    session, user.id, title, body.model_name, body.llm_model
                )
                conversation_id = conv.id
            else:
                conv = await crud.get_conversation(session, body.conversation_id)
                if conv is None or conv.user_id != user.id:
                    raise HTTPException(status_code=404, detail="Conversation not found")
                conversation_id = conv.id

            # Save user message
            await crud.add_message(session, conversation_id, "user", body.question)

            # Fetch last 10 messages for history context
            history_msgs = await crud.get_history(session, conversation_id, limit=10)
            history = [{"role": m.role, "content": m.content} for m in history_msgs]
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    try:
        # Run RAG with history (FAISS retrieval on current question only)
        # Use to_thread since the RAG pipeline is synchronous/blocking
        answer = await asyncio.wait_for(
            asyncio.to_thread(
                rag.faiss_chain_product_based_with_history,
                question=body.question, history=history, initial_k=10, final_k=3,
            ),
            timeout=120,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="RAG query timed out after 120s")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {e}")

    # Save assistant message
    async with async_session() as session:
        await crud.add_message(session, conversation_id, "assistant", answer)

    return AskResponse(answer=answer, conversation_id=conversation_id)


# ── Conversation endpoints ─────────────────────────────────────────

@app.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(user: User = Depends(get_current_user)):
    async with async_session() as session:
        convs = await crud.get_conversations_for_user(session, user.id)
        return convs


@app.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: uuid.UUID, user: User = Depends(get_current_user)
):
    async with async_session() as session:
        conv = await crud.get_conversation(session, conversation_id)
        if conv is None or conv.user_id != user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv


# ── Model listing ──────────────────────────────────────────────────

@app.get("/llm-models")
async def list_llm_models():
    return {"models": LLM_MODELS}


# ── Health check ───────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "ok"}
