"""Streamlit UI for MIL anomaly demo."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from ..demo.coord import run_pipeline
from ..demo.incidents import IncidentDAO
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

st.set_page_config(page_title="Surveillance MIL Demo", layout="wide")
st.title("Surveillance-TF Demo")


def _incident_table(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame(columns=["id", "severity", "mean_conf", "start_ts", "end_ts", "clip_path", "report_path"])
    with IncidentDAO(db_path) as dao:
        records = dao.recent(limit=50)
    data = [
        {
            "id": rec.id,
            "severity": rec.severity,
            "mean_conf": rec.mean_conf,
            "start_ts": rec.start_ts,
            "end_ts": rec.end_ts,
            "clip_path": rec.clip_path,
            "report_path": rec.report_path,
        }
        for rec in records
    ]
    return pd.DataFrame(data)


st.sidebar.header("Configuration")
default_ckpt = Path("models/movinet/ckpt_best")
default_config = Path("configs/thresholds.yaml")
default_video = Path("surveillance_tf/data/dcsass/sample/*.mp4")
default_db = Path("outputs/demo/incidents.sqlite")
default_out = Path("outputs/demo")

ckpt_path = Path(st.sidebar.text_input("Checkpoint path", value=str(default_ckpt)))
config_path = Path(st.sidebar.text_input("Threshold config", value=str(default_config)))
video_path = Path(st.sidebar.text_input("Video path", value=str(default_video)))
fps = float(st.sidebar.number_input("Video FPS", value=25.0, min_value=1.0))
db_path = Path(st.sidebar.text_input("Incident DB", value=str(default_db)))
out_dir = Path(st.sidebar.text_input("Output directory", value=str(default_out)))
seed = int(st.sidebar.number_input("Random seed", value=1337))

if st.sidebar.button("Run Inference"):
    with st.spinner("Processing video..."):
        try:
            incidents = run_pipeline(
                video=video_path,
                ckpt=ckpt_path,
                config_path=config_path,
                fps=fps,
                db_path=db_path,
                out_dir=out_dir,
                seed=seed,
            )
            st.success(f"Completed. Logged {len([i for i in incidents if i.db_id])} incidents.")
        except FileNotFoundError as exc:
            st.error(str(exc))
        except RuntimeError as exc:
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover - UI safety
            st.error(f"Unexpected error: {exc}")

st.subheader("Recent Incidents")
incidents_df = _incident_table(db_path)
if incidents_df.empty:
    st.info("No incidents recorded yet. Configure paths and click 'Run Inference'.")
else:
    st.dataframe(incidents_df, use_container_width=True)
    if "report_path" in incidents_df.columns:
        latest = incidents_df.dropna(subset=["report_path"]).head(1)
        if not latest.empty:
            report_file = Path(latest.iloc[0]["report_path"])
            if report_file.exists():
                st.download_button(
                    "Download latest report",
                    data=report_file.read_bytes(),
                    file_name=report_file.name,
                )
            else:
                st.warning(f"Latest report missing on disk: {report_file}")

st.caption("Ensure the checkpoint and dataset paths are populated as described in the README before running the demo.")
