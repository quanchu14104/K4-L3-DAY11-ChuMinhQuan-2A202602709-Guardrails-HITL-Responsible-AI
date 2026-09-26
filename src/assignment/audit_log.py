"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp keyed by request_id/user_id."""
        import time
        req_id = request_id or f"{user_id}_{len(self.logs)}_{time.time()}"
        self._open[req_id] = {
            "user_id": user_id,
            "text": text,
            "start_time": time.time(),
            "timestamp": utc_now_iso(),
            "request_id": req_id,
        }
        return req_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs."""
        import time
        open_entry = self._open.pop(request_id, None) if request_id else None
        if open_entry is None:
            for k, v in list(self._open.items()):
                if v.get("user_id") == user_id:
                    open_entry = self._open.pop(k)
                    break

        now = time.time()
        start = open_entry.get("start_time", now) if open_entry else now
        latency = round(now - start, 4)
        input_text = open_entry.get("text", "") if open_entry else ""
        timestamp = open_entry.get("timestamp", utc_now_iso()) if open_entry else utc_now_iso()

        entry = {
            "timestamp": timestamp,
            "request_id": request_id or (open_entry.get("request_id") if open_entry else None),
            "user_id": user_id,
            "input": input_text,
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency_seconds": latency,
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        target_path = Path(filepath or default_audit_log_path())
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8")
        return target_path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
