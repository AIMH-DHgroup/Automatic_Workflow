from __future__ import annotations

from automatic_workflow.narrative import fill_missing_titles, serialize_story_text


def test_fill_missing_titles_uses_event_fallbacks() -> None:
    assert fill_missing_titles(["First"], ["A", "B"]) == ["First", "event-2"]


def test_serialize_story_text_outputs_title_description_csv() -> None:
    csv_text = serialize_story_text(["Paragraph one", "Paragraph two"], ["One", "Two"])
    assert csv_text == "title,description\nOne,Paragraph one\nTwo,Paragraph two\n"
