from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2))


def write_jsonl(path: str | Path, rows: Iterable[BaseModel | dict]) -> None:
    out_path = Path(path)
    with out_path.open("w") as handle:
        for row in rows:
            if isinstance(row, BaseModel):
                payload = row.model_dump(mode="python")
            else:
                payload = row
            handle.write(json.dumps(payload) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
