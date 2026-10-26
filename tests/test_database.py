from __future__ import annotations

from pathlib import Path

from app.core.database import Database


def test_chat_and_message_persistence(tmp_path: Path) -> None:
    path = tmp_path / "chats.db"
    db = Database(path)
    chat = db.create_chat(
        title="First chat",
        model="local/foo",
        provider="",
        params={"temperature": 0.5},
        system_prompt="Be brief.",
    )
    user = db.add_message(chat.id, "user", "Hello")
    reply = db.add_message(chat.id, "assistant", "Hi there", {"finish_reason": "stop"})
    db.close()

    reopened = Database(path)
    loaded = reopened.get_chat(chat.id)
    assert loaded is not None
    assert loaded.title == "First chat"
    assert loaded.params == {"temperature": 0.5}
    assert loaded.system_prompt == "Be brief."
    messages = reopened.list_messages(chat.id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "Hello"
    assert messages[1].meta == {"finish_reason": "stop"}
    reopened.close()


def test_messages_are_ordered_even_with_same_timestamp(tmp_path: Path) -> None:
    db = Database(tmp_path / "chats.db")
    chat = db.create_chat()
    for index in range(5):
        db.add_message(chat.id, "user", f"message {index}")
    contents = [m.content for m in db.list_messages(chat.id)]
    assert contents == [f"message {index}" for index in range(5)]
    db.close()


def test_update_message_and_chat(tmp_path: Path) -> None:
    db = Database(tmp_path / "chats.db")
    chat = db.create_chat()
    message = db.add_message(chat.id, "assistant", "partial")
    db.update_message(message.id, "partial answer", {"tokens": 3})
    db.update_chat(chat.id, title="Renamed", model="gpt-4o-mini")

    stored = db.list_messages(chat.id)[0]
    assert stored.content == "partial answer"
    assert stored.meta == {"tokens": 3}
    assert db.get_chat(chat.id).title == "Renamed"
    assert db.get_chat(chat.id).model == "gpt-4o-mini"
    db.close()


def test_delete_chat_cascades_messages(tmp_path: Path) -> None:
    db = Database(tmp_path / "chats.db")
    chat = db.create_chat()
    db.add_message(chat.id, "user", "hello")
    db.delete_chat(chat.id)
    assert db.get_chat(chat.id) is None
    assert db.list_messages(chat.id) == []
    assert db.stats()["messages"] == 0
    db.close()


def test_clear_history_and_stats(tmp_path: Path) -> None:
    db = Database(tmp_path / "chats.db")
    for _ in range(3):
        chat = db.create_chat()
        db.add_message(chat.id, "user", "x")
    stats = db.stats()
    assert stats["chats"] == 3
    assert stats["messages"] == 3
    db.clear_history()
    assert db.list_chats() == []
    assert db.stats()["messages"] == 0
    db.close()


def test_chats_are_sorted_by_recent_activity(tmp_path: Path) -> None:
    db = Database(tmp_path / "chats.db")
    first = db.create_chat("First")
    second = db.create_chat("Second")
    db.update_chat(first.id, title="First updated")
    chats = db.list_chats()
    assert chats[0].id == first.id
    assert {chat.id for chat in chats} == {first.id, second.id}
    db.close()
