"""Chainlit entrypoint for the AuraWealth client chat.

Run with:  python -m chainlit run app/main.py -w
"""

import chainlit as cl

from app import prompts
from app.conversation import Conversation
from app.identity import profile_label, resolve_client_id, signed_in_email
from app.llm import stream_chat
from app.portfolio import money, summarise
from app.seed import BOOK, DEFAULT_CLIENT_ID, get_client

STARTERS = [
    cl.Starter(label="My net worth", message="What's my net worth right now?"),
    cl.Starter(label="Goal progress", message="Am I on track for my goals?"),
    cl.Starter(label="Portfolio mix", message="How is my portfolio allocated?"),
    cl.Starter(label="Biggest risk", message="What's the biggest risk in my portfolio?"),
]


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
    operator = signed_in_email(environ)
    client_id = resolve_client_id(environ, cl.user_session.get("chat_profile"))
    client = get_client(client_id)

    cl.user_session.set("client", client)
    cl.user_session.set(
        "conversation",
        Conversation(system_prompt=prompts.client_advisor(summarise(client))),
    )

    signed_in = f"\n\nSigned in as `{operator}`." if operator else ""
    await cl.Message(
        content=(
            f"**Welcome back, {client.name.split()[0]}.**\n\n"
            f"Net worth today: **{money(client.net_worth)}** "
            f"across {len(client.accounts)} accounts, tracking "
            f"{len(client.goals)} goals."
            f"{signed_in}"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    conversation: Conversation = cl.user_session.get("conversation")
    conversation.add_user(message.content)

    reply = cl.Message(content="")
    await reply.send()

    chunks: list[str] = []
    async for delta in stream_chat(conversation.to_payload()):
        chunks.append(delta)
        await reply.stream_token(delta)
    await reply.update()

    # Record the assistant turn so the next message sees the full history.
    conversation.add_assistant("".join(chunks))
