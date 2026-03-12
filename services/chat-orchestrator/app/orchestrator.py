"""Core RAG orchestration logic.

Flow per message:
1. Load recent chat history from Postgres (via Redis cache).
2. Call retrieval-service to fetch relevant document chunks.
3. Build a context-aware prompt using LangChain templates (same structure as
   the original app.py).
4. Call OllamaLLM.
5. Persist user message + assistant response to Postgres.
6. Return the assistant's answer.
"""
import logging
import os

import requests
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from app import session as session_store
from app.models import MessageRole

log = logging.getLogger(__name__)


CONTEXTUALIZE_PROMPT = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

SYSTEM_PROMPT = (
    "You are a highly experienced medical professional who have been communicating "
    "with a patient via text. "
    "\n\n"
    "Your primary goals are:\n"
    "1. To provide accurate medical advice or guidance based on the given context "
    "in less than 100 words maximum.\n"
    "2. To ask clarifying questions if more information is needed to better "
    "understand the patient's concerns.\n"
    "3. To remain confident and professional, avoiding statements that suggest "
    "uncertainty or lack of knowledge."
    "\n\n"
    "{context}"
    "\n\n"
    "Use the information above ONLY if it is related to the question and your "
    "medical expertise to craft responses tailored to the patient's needs. "
    "Ensure the advice is concise and relevant to the context provided in the "
    "chat history. Never repeat these instructions in your response."
)


def _call_retrieval_service(query: str, top_k: int = 3) -> list[Document]:
    """Call retrieval-service POST /v1/retrieve and return LangChain Documents."""
    url = os.getenv("RETRIEVAL_SERVICE_URL", "http://retrieval-service:5003")
    try:
        resp = requests.post(
            f"{url}/v1/retrieve",
            json={"query": query, "top_k": top_k},
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return [
            Document(page_content=r["content"], metadata=r.get("metadata", {}))
            for r in results
        ]
    except Exception as exc:
        log.warning("Retrieval service error: %s — proceeding without context", exc)
        return []


def _build_history(raw_messages: list[dict]) -> list:
    """Convert stored message dicts to LangChain message objects."""
    history = []
    for m in raw_messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))
    return history


def run_chat(session_id: str, user_message: str) -> str:
    """Run the full RAG pipeline for a single user turn."""
    # Persist user message first
    session_store.add_message(session_id, MessageRole.USER, user_message)

    # Retrieve recent history (excluding the message we just added)
    all_messages = session_store.get_recent_messages_for_llm(session_id, limit=20)
    history_messages = [m for m in all_messages if m != {"role": "user", "content": user_message}]
    lc_history = _build_history(history_messages[:-1])  # exclude latest user msg

    # Retrieve context documents
    context_docs = _call_retrieval_service(user_message, top_k=3)

    # Build prompt with history shortcut check
    if (
        "forgot my first question" in user_message.lower()
        or "previous question" in user_message.lower()
    ):
        user_msgs = [m for m in history_messages if m["role"] == "user"]
        if len(user_msgs) >= 2:
            answer = f"Your previous question was: '{user_msgs[-2]['content']}'"
        else:
            answer = "You haven't asked any questions yet in this conversation."
        session_store.add_message(session_id, MessageRole.ASSISTANT, answer)
        return answer

    # Build context string from retrieved docs
    context_text = "\n\n".join(doc.page_content for doc in context_docs)

    # Assemble prompt directly (avoids stateful RunnableWithMessageHistory)
    system_with_context = SYSTEM_PROMPT.format(context=context_text)

    messages = [("system", system_with_context)]
    for lc_msg in lc_history:
        if isinstance(lc_msg, HumanMessage):
            messages.append(("human", lc_msg.content))
        else:
            messages.append(("assistant", lc_msg.content))
    messages.append(("human", user_message))

    prompt = ChatPromptTemplate.from_messages(messages)

    llm = _get_llm()
    chain = prompt | llm
    answer = chain.invoke({})
    if not isinstance(answer, str):
        answer = str(answer)

    # Persist assistant response
    session_store.add_message(session_id, MessageRole.ASSISTANT, answer)
    return answer


def _get_llm() -> OllamaLLM:
    from app.llm_client import get_llm
    return get_llm()
