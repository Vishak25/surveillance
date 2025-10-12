from pathlib import Path

import numpy as np
import pandas as pd

from surveillance_tf.data import dcsass_loader


def _stub_decode(_path, target_size=(224, 224)):
    frame = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    return [frame for _ in range(40)]


def _stub_segments(frames, T, stride):
    return np.zeros((4, T, frames[0].shape[0], frames[0].shape[1], 3), dtype=np.uint8)


def test_dcsass_make_dataset_with_metadata(monkeypatch, tmp_path):
    data_root = tmp_path / "dcsass"
    dataset_dir = data_root / "DCSASS Dataset" / "Abuse"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "clip001.mp4").touch()
    metadata = data_root / "metadata.csv"
    pd.DataFrame({
        "path": ["Abuse/clip001.mp4"],
        "label": ["Abuse"],
        "split": ["train"],
    }).to_csv(metadata, index=False)

    monkeypatch.setattr(dcsass_loader, "decode_video_opencv", _stub_decode)
    monkeypatch.setattr(dcsass_loader, "make_segments", _stub_segments)

    ds = dcsass_loader.make_bag_dataset(data_root, split="train", batch_size=1, image_size=(224, 224))
    segments, label, video_id = next(iter(ds))
    tensor = segments.to_tensor()
    assert tensor.shape[2:] == (224, 224, 3)
    assert int(label.numpy()[0]) == 1  # Abuse treated as abnormal
    assert video_id.numpy()[0].decode("utf-8").endswith("clip001.mp4")
