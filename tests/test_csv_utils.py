from __future__ import annotations

from pathlib import Path

from automatic_workflow.csv_utils import read_csv_rows, write_csv_rows


def test_write_and_read_csv_rows(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "rows.csv"

    write_csv_rows(path, [["a", "b,c"]], quote_all=True)

    assert path.exists()
    assert read_csv_rows(path) == [["a", "b,c"]]
