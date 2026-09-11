"""Multi-turn conversation state.

Kept deliberately free of any Chainlit or HTTP dependency: the agent layer will
need to build, trim and replay these message lists too, and a plain object is
far easier to test than framework session state.
"""

from dataclasses import dataclass, field

Message = dict[str, object]


@dataclass
class Conversation:
    """An ordered message history for a single chat session.

    The system prompt is stored separately from ``turns`` so it can never be
    trimmed away or duplicated when the history is truncated.
    """

    system_prompt: str
    turns: list[Message] = field(default_factory=list)
    max_turns: int = 40

    def add_user(self, content: object) -> None:
        self._append({"role": "user", "content": content})

    def add_assistant(self, content: object) -> None:
        self._append({"role": "assistant", "content": content})

    def _append(self, message: Message) -> None:
        self.turns.append(message)
        self._trim()

    def _trim(self) -> None:
        """Drop the oldest turns once the history exceeds ``max_turns``.

        Trimming from the front keeps the most recent context, and dropping in
        pairs avoids leaving a dangling assistant message as the first turn.
        """
        overflow = len(self.turns) - self.max_turns
        if overflow > 0:
            del self.turns[: overflow + (overflow % 2)]

    def to_payload(self) -> list[Message]:
        """Message list in the shape the serving endpoint expects."""
        return [{"role": "system", "content": self.system_prompt}, *self.turns]
