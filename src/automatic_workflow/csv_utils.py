"""CSV helpers used by the workflow."""

from __future__ import annotations

import csv
import io
from pathlib import Path


def build_csv_text(rows: list[list[str]], delimiter: str = ",") -> str:
    """Serialize rows to CSV text.

    Args:
        rows: Rows to serialize.
        delimiter: CSV delimiter.

    Returns:
        CSV text using `\n` line endings.
    """

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\n")
    writer.writerows(rows)
    return buffer.getvalue()


def read_csv_rows(path: Path, delimiter: str = ",", encoding: str = "utf-8") -> list[list[str]]:
    """Read CSV rows with standard double-quote handling."""

    with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        return list(csv.reader(handle, delimiter=delimiter, quotechar='"', doublequote=True))


def write_csv_rows(
    path: Path,
    rows: list[list[str]],
    delimiter: str = ",",
    encoding: str = "utf-8",
    quote_all: bool = False,
) -> None:
    """Write CSV rows to disk, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    quoting = csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter, quoting=quoting, lineterminator="\n")
        writer.writerows(rows)
