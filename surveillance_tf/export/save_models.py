"""Utilities to export SavedModel bundles."""
from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from ..models.mil_head import MILScoringHead
from ..models.movinet_backbone import build_segment_encoder
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def export_models(args: argparse.Namespace) -> None:
    backbone = build_segment_encoder(input_shape=(args.window, 224, 224, 3), trainable=False, dropout_rate=0.0)
    if isinstance(args.head_units, str):
        head_units = [int(u.strip()) for u in args.head_units.split(",") if u.strip()]
    else:
        head_units = [256, 128]
    head = MILScoringHead(units=head_units)

    export_dir = Path(args.output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    backbone_path = export_dir / "backbone"
    head_path = export_dir / "mil_head"

    tf.saved_model.save(backbone, backbone_path)
    tf.saved_model.save(head, head_path)

    LOGGER.info("Saved backbone to %s and head to %s", backbone_path, head_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/saved_models"))
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--head-units", type=str, default="256,128")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    export_models(args)


if __name__ == "__main__":
    main()
