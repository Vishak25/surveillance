"""Incident reporting helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence
from urllib import request

from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

from ..utils.clip_writer import write_clip_from_frames
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)
TEMPLATE_DIR = Path(__file__).resolve().parent / "report_templates"


def render_report(incident: dict, snapshots: Sequence[Path], out_pdf: str | Path) -> Path:
    """Render a PDF report using Jinja2 + WeasyPrint."""
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("incident.html.j2")
    html = template.render(incident=incident, snapshots=[str(p) for p in snapshots])
    out_path = Path(out_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html).write_pdf(str(out_path))
    LOGGER.info("Report saved to %s", out_path)
    return out_path


def export_clip(frames_with_ts: Iterable, fps: float, out_path: str | Path) -> Path:
    return write_clip_from_frames(frames_with_ts, fps, out_path)


def maybe_notify_discord(message: str, report_path: Path | None = None) -> None:
    webhook = os.environ.get("DISCORD_WEBHOOK")
    if not webhook:
        return
    payload = {"content": message}
    if report_path and report_path.exists():
        payload["embeds"] = [
            {
                "title": "Incident Report",
                "description": str(report_path),
            }
        ]
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(webhook, data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=5) as resp:
            LOGGER.info("Discord webhook status: %s", resp.status)
    except Exception as exc:  # pragma: no cover - network optional
        LOGGER.warning("Failed to notify Discord: %s", exc)


__all__ = ["render_report", "export_clip", "maybe_notify_discord"]
