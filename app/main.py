"""Chainlit entrypoint for the AuraWealth client chat.

Run with:  chainlit run app/main.py -w
"""

import chainlit as cl

from app.conversation import Conversation
from app.llm import stream_chat
from app.prompts import CLIENT_ADVISOR

WELCOME = (
    "**AuraWealth** — your financial GPS.\n\n"
    "Ask me about your net worth, your goals, or anything in your portfolio."
)


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("conversation", Conversation(system_prompt=CLIENT_ADVISOR))
    await cl.Message(content=WELCOME).send()


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
