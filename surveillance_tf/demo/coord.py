"""Runtime anomaly coordination and incident logging pipeline."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum, auto
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import yaml

from surveillance_tf.demo.fusion import fuse_scores, predict_with_ci
from surveillance_tf.demo.incidents import IncidentDAO, IncidentRecord
from surveillance_tf.demo.responder import export_clip, maybe_notify_discord, render_report
from surveillance_tf.utils.logging import get_logger
from surveillance_tf.utils.paths import resolve_dcsass_root
from surveillance_tf.utils.ring_buffer import VideoRingBuffer
from surveillance_tf.utils.seed import set_global_seed

LOGGER = get_logger(__name__)


class IncidentState(Enum):
    OPEN = auto()
    READY = auto()
    ALERTED = auto()
    CLOSED = auto()


@dataclass
class Configuration:
    anomaly_threshold: float
    ci_min_for_alert: float
    cooldown_seconds: float
    window_frames: int
    stride_frames: int
    code_red_classes: Sequence[str]
    code_yellow_classes: Sequence[str]


@dataclass
class Incident:
    incident_id: int
    start_ts: float
    end_ts: float
    mean_conf: float
    ci_low: float
    ci_high: float
    severity: str
    state: IncidentState = IncidentState.OPEN
    exported: bool = False
    history: List[Dict[str, float]] = None
    db_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []

    def update(self, timestamp: float, mean_conf: float, ci_low: float, ci_high: float) -> None:
        self.end_ts = timestamp
        self.mean_conf = max(self.mean_conf, mean_conf)
        self.ci_low = min(self.ci_low, ci_low)
        self.ci_high = max(self.ci_high, ci_high)
        self.history.append(
            {
                "timestamp": timestamp,
                "mean": mean_conf,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )


class IncidentCoordinator:
    def __init__(self, config: Configuration):
        self.config = config
        self.incidents: List[Incident] = []
        self.next_id = 1

    def _severity(self, class_name: str, mean_conf: float) -> str:
        if class_name in self.config.code_red_classes or mean_conf >= self.config.anomaly_threshold + 0.15:
            return "code_red"
        if class_name in self.config.code_yellow_classes or mean_conf >= self.config.anomaly_threshold:
            return "code_yellow"
        return "code_blue"

    def register(self, timestamp: float, mean_conf: float, ci_low: float, ci_high: float, class_name: str) -> Incident:
        for incident in reversed(self.incidents):
            if timestamp - incident.end_ts <= self.config.cooldown_seconds:
                incident.update(timestamp, mean_conf, ci_low, ci_high)
                return incident
        incident = Incident(
            incident_id=self.next_id,
            start_ts=timestamp,
            end_ts=timestamp,
            mean_conf=mean_conf,
            ci_low=ci_low,
            ci_high=ci_high,
            severity=self._severity(class_name, mean_conf),
        )
        incident.update(timestamp, mean_conf, ci_low, ci_high)
        self.incidents.append(incident)
        self.next_id += 1
        return incident

    def advance_state(self, incident: Incident) -> None:
        if incident.state == IncidentState.OPEN and incident.mean_conf >= self.config.anomaly_threshold:
            incident.state = IncidentState.READY
        if incident.state == IncidentState.READY and incident.ci_low >= self.config.ci_min_for_alert:
            incident.state = IncidentState.ALERTED

    def close_stale(self, timestamp: float) -> None:
        for incident in self.incidents:
            if incident.state != IncidentState.CLOSED and timestamp - incident.end_ts > self.config.cooldown_seconds:
                incident.state = IncidentState.CLOSED


def _load_config(path: Path) -> Configuration:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return Configuration(
        anomaly_threshold=float(cfg["anomaly_threshold"]),
        ci_min_for_alert=float(cfg["ci_min_for_alert"]),
        cooldown_seconds=float(cfg["cooldown_seconds"]),
        window_frames=int(cfg["window_frames"]),
        stride_frames=int(cfg["stride_frames"]),
        code_red_classes=tuple(cfg.get("code_red_classes", [])),
        code_yellow_classes=tuple(cfg.get("code_yellow_classes", [])),
    )


def _expand_videos(pattern: str, dataset_root: Path) -> List[Path]:
    pattern_path = Path(pattern)
    candidates: List[str] = []
    if pattern_path.is_absolute():
        candidates.extend(glob(str(pattern_path)))
    else:
        candidates.extend(glob(str(pattern_path)))
        candidates.extend(glob(str((dataset_root / pattern_path).resolve())))
    unique: List[Path] = []
    seen = set()
    for item in candidates:
        normalized = Path(item).resolve()
        if normalized not in seen:
            unique.append(normalized)
            seen.add(normalized)
    return unique


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=None, help="Dataset root (defaults to ./data/dcsass)")
    parser.add_argument("--video", type=str, required=True, help="Video file or glob pattern")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--db", type=Path, default=Path("outputs/demo/incidents.sqlite"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--passes", type=int, default=8, help="MC-dropout passes per window")
    return parser.parse_args(list(argv) if argv is not None else None)


def _export_incident_assets(
    incident: Incident,
    ring_buffer: VideoRingBuffer,
    video_path: Path,
    out_dir: Path,
    fps: float,
) -> tuple[Optional[Path], Optional[Path], List[Path]]:
    import cv2

    clip_dir = out_dir / "clips"
    report_dir = out_dir / "reports"
    snap_dir = out_dir / "snapshots"
    clip_path: Optional[Path] = None
    report_path: Optional[Path] = None
    snapshots: List[Path] = []
    try:
        clip_start = max(incident.start_ts - 2.0, 0.0)
        clip_end = incident.end_ts + 2.0
        clip_path = ring_buffer.export_clip(clip_start, clip_end, clip_dir / f"incident_{incident.incident_id}.mp4", fps=fps)
        frames = list(ring_buffer.iter_window(clip_start, clip_end))
        snapshots = []
        snap_dir.mkdir(parents=True, exist_ok=True)
        if frames:
            import numpy as np

            indices = np.linspace(0, len(frames) - 1, num=min(3, len(frames)), dtype=int)
            for idx in indices:
                frame, ts = frames[idx]
                filename = snap_dir / f"incident_{incident.incident_id}_{ts:.2f}.jpg"
                cv2.imwrite(str(filename), frame)
                snapshots.append(filename)
        report_payload = {
            "incident": incident,
            "video_path": str(video_path),
        }
        report_path = render_report(report_payload, snapshots, report_dir / f"incident_{incident.incident_id}.pdf")
    except Exception as exc:  # pragma: no cover - IO failure
        LOGGER.error("Failed to export incident assets: %s", exc)
        return clip_path, report_path, snapshots
    return clip_path, report_path, snapshots


def run_pipeline(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np
    import tensorflow as tf

    dataset_root = resolve_dcsass_root(args.data_root)
    video_candidates = _expand_videos(args.video, dataset_root)
    if not video_candidates:
        LOGGER.error("No video files matched pattern '%s'.", args.video)
        return
    video_path = video_candidates[0]
    LOGGER.info("Processing video %s", video_path)

    config = _load_config(args.config)
    coordinator = IncidentCoordinator(config)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    dao = IncidentDAO(args.db)

    clip_model = tf.keras.models.load_model(str(args.ckpt))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        LOGGER.error("Failed to open %s", video_path)
        return

    ring_buffer = VideoRingBuffer(seconds=int(config.cooldown_seconds), fps=args.fps)
    frames_window: List[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_idx / args.fps
        ring_buffer.push(frame, timestamp)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, tuple(args.image_size))
        frames_window.append(resized)
        if len(frames_window) > config.window_frames:
            frames_window.pop(0)

        if len(frames_window) == config.window_frames and frame_idx % config.stride_frames == 0:
            window_arr = np.array(frames_window, dtype=np.float32) / 255.0
            segment = tf.convert_to_tensor(window_arr[None, ...], dtype=tf.float32)
            mean_score, std_score = predict_with_ci(clip_model, segment, passes=args.passes)
            ci_low = max(0.0, mean_score - 1.96 * std_score)
            ci_high = min(1.0, mean_score + 1.96 * std_score)
            fused = fuse_scores(mean_score)

            if fused >= config.anomaly_threshold:
                incident = coordinator.register(timestamp, fused, ci_low, ci_high, "Abnormal")
                coordinator.advance_state(incident)
                if incident.state == IncidentState.ALERTED and not incident.exported:
                    clip_path, report_path, snapshots = _export_incident_assets(
                        incident,
                        ring_buffer,
                        video_path,
                        args.out,
                        fps=args.fps,
                    )
                    record = IncidentRecord(
                        start_ts=incident.start_ts,
                        end_ts=incident.end_ts,
                        severity=incident.severity,
                        class_name="Abnormal",
                        mean_conf=incident.mean_conf,
                        ci_low=incident.ci_low,
                        ci_high=incident.ci_high,
                        video_path=str(video_path),
                        clip_path=str(clip_path) if clip_path else None,
                        report_path=str(report_path) if report_path else None,
                    )
                    incident.db_id = dao.log(record)
                    maybe_notify_discord(
                        f"Incident {incident.incident_id} ({incident.severity}) score={incident.mean_conf:.2f}",
                        Path(record.report_path) if record.report_path else None,
                    )
                    incident.exported = True
            coordinator.close_stale(timestamp)
        frame_idx += 1

    cap.release()
    dao.close()
    LOGGER.info("Processed %d frames. Incidents logged: %d", frame_idx, len([i for i in coordinator.incidents if i.db_id]))


if __name__ == "__main__":
    args = parse_args()
    set_global_seed(args.seed)
    run_pipeline(args)
