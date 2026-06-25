from __future__ import annotations

from automatic_workflow.llm_json import extract_first_json_object, parse_entities_response


def test_extract_first_json_object_repairs_trailing_commas() -> None:
    result = extract_first_json_object(
        '```json\n{"entities": [{"wikipedia_title": "Rome",}],}\n```'
    )
    assert result is not None
    assert "Rome" in result


def test_parse_entities_response_filters_invalid_items() -> None:
    parsed = parse_entities_response(
        '{"entities": [{"entity_in_the_text": "Rome", "wikipedia_title": "Rome"}, '
        '{"entity_in_the_text": "Missing"}]}'
    )
    assert parsed == [{"entity_in_the_text": "Rome", "wikipedia_title": "Rome"}]


def test_parse_entities_response_returns_empty_for_non_json() -> None:
    assert parse_entities_response("not json") == []
