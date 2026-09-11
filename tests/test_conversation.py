from app.conversation import Conversation

SYSTEM = "you are a test advisor"


def test_payload_keeps_system_first_and_preserves_turn_order():
    """Multi-turn history must replay in order, behind a single system prompt."""
    convo = Conversation(system_prompt=SYSTEM)
    convo.add_user("what is my net worth?")
    convo.add_assistant("about $420,000")
    convo.add_user("and how much of that is cash?")

    payload = convo.to_payload()

    assert payload[0] == {"role": "system", "content": SYSTEM}
    assert [m["role"] for m in payload] == ["system", "user", "assistant", "user"]
    assert payload[3]["content"] == "and how much of that is cash?"
    # The system prompt lives outside `turns`, so it cannot be duplicated.
    assert all(m["role"] != "system" for m in convo.turns)


def test_trimming_bounds_history_without_losing_the_system_prompt():
    """Long sessions stay under the cap; the system prompt always survives."""
    convo = Conversation(system_prompt=SYSTEM, max_turns=4)
    for i in range(6):
        convo.add_user(f"question {i}")
        convo.add_assistant(f"answer {i}")

    assert len(convo.turns) <= 4
    # Oldest turns are dropped, most recent are kept.
    assert convo.turns[-1]["content"] == "answer 5"
    assert all("question 0" != m["content"] for m in convo.turns)
    assert convo.to_payload()[0]["content"] == SYSTEM


def test_trimming_drops_whole_exchanges_so_history_never_starts_mid_turn():
    """A history beginning with an assistant reply confuses the model."""
    convo = Conversation(system_prompt=SYSTEM, max_turns=2)
    for i in range(5):
        convo.add_user(f"q{i}")
        convo.add_assistant(f"a{i}")

    assert convo.turns[0]["role"] == "user"


def test_multimodal_content_survives_the_history():
    """Vision turns carry a content *list*, not a string — don't stringify it."""
    convo = Conversation(system_prompt=SYSTEM)
    blocks = [
        {"type": "text", "text": "what does this statement show?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    convo.add_user(blocks)

    assert convo.to_payload()[1]["content"] == blocks
