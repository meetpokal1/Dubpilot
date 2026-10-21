"""Durable stage checkpoints for long-running dubbing jobs.

The checkpoint is intentionally JSON so it is inspectable and easy to recover.
Only stage metadata and paths are stored here; transcript/audio artifacts stay
inside the configured work directory.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGE_ORDER = (
    "download",
    "transcribe",
    "translate",
    "diarize",
    "synthesize",
    "remix",
    "complete",
)


class PipelineCheckpoint:
    def __init__(
        self,
        path: str | os.PathLike,
        config: dict[str, Any],
        resume: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if resume and self.path.exists():
            try:
                stored = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Checkpoint {self.path} is unreadable. Move it aside or start without --resume."
                ) from exc
            if stored.get("config") != config:
                raise ValueError(
                    "The saved checkpoint belongs to different URL/model/voice/output settings. "
                    "Use the original settings with --resume, choose a new --work-dir, or start without --resume."
                )
            self.data = stored
        else:
            self.data = {
                "version": 1,
                "config": config,
                "completed_stages": [],
                "artifacts": {},
                "updated_at": self._now(),
            }
            self.save()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def save(self) -> None:
        self.data["updated_at"] = self._now()
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def is_complete(self, stage: str) -> bool:
        return stage in self.data.get("completed_stages", [])

    def artifact(self, stage: str) -> dict[str, Any]:
        return dict(self.data.get("artifacts", {}).get(stage, {}))

    def mark_complete(self, stage: str, **artifacts: Any) -> None:
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        self.data.setdefault("artifacts", {})[stage] = artifacts
        completed = self.data.setdefault("completed_stages", [])
        if stage not in completed:
            completed.append(stage)
        completed.sort(key=STAGE_ORDER.index)
        self.save()

    def invalidate_from(self, stage: str) -> None:
        """Forget a stage and all downstream stage markers before recomputing."""
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        start = STAGE_ORDER.index(stage)
        invalidated = set(STAGE_ORDER[start:])
        self.data["completed_stages"] = [
            item for item in self.data.get("completed_stages", [])
            if item not in invalidated
        ]
        artifacts = self.data.setdefault("artifacts", {})
        for item in invalidated:
            artifacts.pop(item, None)
        self.save()
