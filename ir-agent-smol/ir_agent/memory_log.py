import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class IRTaskMemoryLog:
    def __init__(self, log_path: str, max_items: int = 5000) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_items = max_items

    def append(self, entry: dict[str, Any]) -> None:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **entry,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []

        rows: list[dict[str, Any]] = []
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if len(rows) > self.max_items:
            return rows[-self.max_items :]
        return rows

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.read_all()
        return rows[-max(1, limit) :]

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        q = (query or "").strip().lower()
        if not q:
            return self.recent(limit)

        scored: list[tuple[int, dict[str, Any]]] = []
        for item in self.read_all():
            hay = " ".join(
                [
                    str(item.get("prompt", "")),
                    str(item.get("route", "")),
                    str(item.get("tool", "")),
                    str(item.get("image_path", "")),
                    str(item.get("notes", "")),
                ]
            ).lower()
            score = hay.count(q)
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[: max(1, limit)]]
