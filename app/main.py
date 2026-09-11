"""Chainlit entrypoint for the AuraWealth client chat.

Run with:  python -m chainlit run app/main.py -w
"""

import asyncio
import logging
from datetime import date

import chainlit as cl

from app import prompts
from app.agents.pipeline import format_review, run_review
from app.conversation import Conversation
from app.identity import profile_label, resolve_client_id
from app.llm import stream_chat
from app.portfolio import money, summarise
from app.rag.retriever import Retriever, format_context
from app.seed import BOOK, DEFAULT_CLIENT_ID, get_client

log = logging.getLogger(__name__)

REVIEW_MESSAGE = "Run a full portfolio review"
REVIEW_TRIGGERS = ("full portfolio review", "full review", "review my portfolio")

STARTERS = [
    cl.Starter(label="My net worth", message="What's my net worth right now?"),
    cl.Starter(label="Goal progress", message="Am I on track for my goals?"),
    cl.Starter(label="Portfolio mix", message="How is my portfolio allocated?"),
    cl.Starter(label="Full review", message=REVIEW_MESSAGE),
]

# One retriever for the whole app. The corpus is identical for every user and
# embedding it costs a round trip per batch, so building it per session would
# make every new chat wait for work already done.
_retriever: Retriever | None = None
_retriever_lock = asyncio.Lock()


async def get_retriever() -> Retriever:
    global _retriever
    async with _retriever_lock:
        if _retriever is None:
            _retriever = await Retriever.build()
            log.info(
                "retriever ready: %d chunks, semantic=%s",
                len(_retriever.chunks),
                _retriever.semantic_available,
            )
    return _retriever


@cl.set_chat_profiles
async def chat_profiles(current_user=None, language=None):
    """One profile per client, so a demo can switch portfolios in the UI."""
    return [
        cl.ChatProfile(
            name=profile_label(client),
            markdown_description=(
                f"**{client.segment}**, age {client.age}, "
                f"{client.risk_profile} risk. "
                f"Net worth {money(client.net_worth)}."
            ),
            starters=STARTERS,
            default=(cid == DEFAULT_CLIENT_ID),
        )
        for cid, client in BOOK.items()
    ]


def _session_environ() -> dict | None:
    """WSGI environ for this session, carrying the Databricks SSO headers.

    Guarded: the attribute is absent for non-websocket clients, and a missing
    identity should downgrade to the default client rather than break the chat.
    """
    try:
        return getattr(cl.context.session, "environ", None)
    except Exception:
        return None


@cl.on_chat_start
async def on_chat_start() -> None:
    environ = _session_environ()
    client_id = resolve_client_id(environ, cl.user_session.get("chat_profile"))
    client = get_client(client_id)

    cl.user_session.set("client", client)
    cl.user_session.set(
        "conversation",
        Conversation(system_prompt=prompts.client_advisor(summarise(client))),
    )

    await cl.Message(
        content=(
            f"**Welcome back, {client.name.split()[0]}.**\n\n"
            f"Net worth today: **{money(client.net_worth)}** "
            f"across {len(client.accounts)} accounts, tracking "
            f"{len(client.goals)} goals."
        )
    ).send()


# --- the sequential review -------------------------------------------------


async def _run_review(client) -> None:
    """Run Analyst -> Risk -> Advisor, rendering each stage as a nested step."""
    retriever = await get_retriever()

    async def report(stage: str, status: str, detail: str) -> None:
        async with cl.Step(name=stage, type="llm") as step:
            step.output = f"**{status}**\n\n{detail}"

    review = await run_review(client, retriever, today=date.today(), report=report)
    await cl.Message(content=format_review(review)).send()

    # Keep the review in history so follow-up questions can refer to it.
    conversation: Conversation = cl.user_session.get("conversation")
    conversation.add_assistant(format_review(review))


# --- ordinary chat, grounded in retrieval ----------------------------------


async def _answer(client, question: str) -> None:
    conversation: Conversation = cl.user_session.get("conversation")
    retriever = await get_retriever()

    async with cl.Step(name="Retrieval", type="retrieval") as step:
        chunks = await retriever.search(question, client=client, k=4)
        step.output = (
            "\n".join(
                f"**S{i}** {c.meta.title} — score {c.scores['reranked']:.3f}"
                for i, c in enumerate(chunks, start=1)
            )
            or "No sources matched."
        )

    payload = conversation.to_payload()
    if chunks:
        # Inserted immediately before the user's turn rather than folded into
        # the system prompt: the sources are specific to this question, and
        # leaving them in the system prompt would keep them in force for every
        # later turn as though they had been retrieved for that one too.
        payload.insert(
            -1,
            {
                "role": "system",
                "content": (
                    "Sources retrieved for this question. Cite the ones you use "
                    "as [S1], [S2] and so on. If they do not answer the question, "
                    "say so rather than filling the gap from memory. Never rely "
                    "on a source marked withdrawn or superseded.\n\n"
                    f"{format_context(chunks)}"
                ),
            },
        )

    reply = cl.Message(content="")
    await reply.send()

    parts: list[str] = []
    async for delta in stream_chat(payload):
        parts.append(delta)
        await reply.stream_token(delta)
    await reply.update()

    conversation.add_assistant("".join(parts))


@cl.on_message
async def on_message(message: cl.Message) -> None:
    client = cl.user_session.get("client")
    conversation: Conversation = cl.user_session.get("conversation")
    conversation.add_user(message.content)

    if any(trigger in message.content.lower() for trigger in REVIEW_TRIGGERS):
        await _run_review(client)
    else:
        await _answer(client, message.content)
