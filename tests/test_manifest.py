from __future__ import annotations

"""Tests für Manifest-Generierung: T-08 (Timestamp-Korrektheit)."""

import json
import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import src.extract as mod
from src.extract import _write_manifest
from src.utils import frame_to_timestamp


@pytest.fixture(autouse=True)
def _patch_log(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "log", logging.getLogger("test"))


def _make_frame(directory: Path, name: str) -> Path:
    arr = np.full((32, 32, 3), 128, dtype=np.uint8)
    path = directory / name
    Image.fromarray(arr).save(path, "JPEG")
    return path


class TestWriteManifest:
    """T-08: Manifest-Timestamps basieren auf orig_raw_paths, nicht auf unique Dateinamen."""

    def test_timestamps_from_raw_paths(self, tmp_path: Path) -> None:
        """Timestamps kommen vom Raw-Frame-Pfad, nicht vom umnummerierten Unique-Frame."""
        frames_raw_dir = tmp_path / "frames_raw"
        frames_raw_dir.mkdir()
        frames_unique_dir = tmp_path / "frames_unique"
        frames_unique_dir.mkdir()

        # Raw-Frames: frame_00010, frame_00050, frame_00100
        raw_10 = _make_frame(frames_raw_dir, "frame_00010.jpg")
        raw_50 = _make_frame(frames_raw_dir, "frame_00050.jpg")
        raw_100 = _make_frame(frames_raw_dir, "frame_00100.jpg")

        # Unique-Frames: neu nummeriert frame_00001, 00002, 00003
        u1 = _make_frame(frames_unique_dir, "frame_00001.jpg")
        u2 = _make_frame(frames_unique_dir, "frame_00002.jpg")
        u3 = _make_frame(frames_unique_dir, "frame_00003.jpg")

        _write_manifest(
            project_dir=tmp_path,
            unique_frames=[u1, u2, u3],
            orig_raw_paths=[raw_10, raw_50, raw_100],
            fps=2.0,
            threshold_used=20,
            auto_threshold=True,
        )

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        entries = manifest["frames"]

        assert entries[0]["original_frame_num"] == 10
        assert entries[1]["original_frame_num"] == 50
        assert entries[2]["original_frame_num"] == 100

        assert entries[0]["timestamp_video"] == frame_to_timestamp(10, 2.0)
        assert entries[1]["timestamp_video"] == frame_to_timestamp(50, 2.0)
        assert entries[2]["timestamp_video"] == frame_to_timestamp(100, 2.0)

    def test_not_using_unique_frame_number(self, tmp_path: Path) -> None:
        """frame_00001.jpg mit raw-Quelle frame_00050.jpg darf NICHT Timestamp von Frame 1 haben."""
        frames_raw_dir = tmp_path / "frames_raw"
        frames_raw_dir.mkdir()
        frames_unique_dir = tmp_path / "frames_unique"
        frames_unique_dir.mkdir()

        raw_50 = _make_frame(frames_raw_dir, "frame_00050.jpg")
        u1 = _make_frame(frames_unique_dir, "frame_00001.jpg")

        _write_manifest(
            project_dir=tmp_path,
            unique_frames=[u1],
            orig_raw_paths=[raw_50],
            fps=2.0,
            threshold_used=20,
            auto_threshold=False,
        )

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        entry = manifest["frames"][0]

        # Muss Timestamp von Frame 50 haben, nicht von Frame 1
        wrong_ts = frame_to_timestamp(1, 2.0)
        correct_ts = frame_to_timestamp(50, 2.0)
        assert entry["timestamp_video"] == correct_ts
        assert entry["timestamp_video"] != wrong_ts

    def test_meta_contains_pass_counts(self, tmp_path: Path) -> None:
        """Manifest-Meta enthält pass1_count und pass2_count wenn angegeben."""
        frames_raw_dir = tmp_path / "frames_raw"
        frames_raw_dir.mkdir()
        frames_unique_dir = tmp_path / "frames_unique"
        frames_unique_dir.mkdir()

        raw = _make_frame(frames_raw_dir, "frame_00001.jpg")
        u = _make_frame(frames_unique_dir, "frame_00001.jpg")

        _write_manifest(
            project_dir=tmp_path,
            unique_frames=[u],
            orig_raw_paths=[raw],
            fps=1.0,
            threshold_used=20,
            auto_threshold=True,
            pass1_count=49,
            pass2_count=42,
            video_type="slideshow",
        )

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        meta = manifest["meta"]
        assert meta["pass1_count"] == 49
        assert meta["pass2_count"] == 42
        assert meta["video_type"] == "slideshow"
        assert meta["fps"] == 1.0
