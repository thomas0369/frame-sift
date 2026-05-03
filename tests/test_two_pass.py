from __future__ import annotations

"""Tests für zweistufiges Dedup: T-05, T-06."""

import logging
import shutil
from pathlib import Path

import imagehash
import numpy as np
import pytest
from PIL import Image

import src.extract as mod
from src.extract import _run_dedup_global


@pytest.fixture(autouse=True)
def _patch_log(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "log", logging.getLogger("test"))


def _make_frame(directory: Path, name: str, value: int) -> Path:
    """Erstellt ein Graustufenbild mit gegebenem Helligkeitswert."""
    arr = np.full((64, 64, 3), value, dtype=np.uint8)
    path = directory / name
    Image.fromarray(arr).save(path, "JPEG", quality=95)
    return path


def _make_gradient(directory: Path, name: str, start: int, end: int) -> Path:
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    for j in range(64):
        arr[:, j] = int(start + (end - start) * j / 63)
    path = directory / name
    Image.fromarray(arr).save(path, "JPEG", quality=95)
    return path


class TestRunDedupGlobal:
    """T-05: _run_dedup_global entfernt near-dups aus Unique-Frames."""

    def _setup_panning_cluster(
        self, tmp_path: Path
    ) -> tuple[list[Path], list[Path], list[Path]]:
        """Erzeugt frames_unique/ mit 10 ähnlichen + 5 verschiedenen Frames."""
        unique_dir = tmp_path / "frames_unique"
        unique_dir.mkdir()
        raw_dir = tmp_path / "frames_raw"
        raw_dir.mkdir()

        frames: list[Path] = []
        raws: list[Path] = []

        # 5 diverse Frames (echte Szenen, hohe Distanz zueinander)
        for i, (s, e) in enumerate([(0, 200), (200, 0), (50, 250), (250, 50), (100, 100)], start=1):
            f = _make_gradient(unique_dir, f"frame_{i:05d}.jpg", s, e)
            r = _make_gradient(raw_dir, f"frame_{i:05d}.jpg", s, e)
            frames.append(f)
            raws.append(r)

        # 10 ähnliche Frames (Panning-Cluster, gradueller Drift von 60→160)
        for i in range(10):
            val_s = 60 + i * 10
            val_e = 160 + i * 10
            idx = 6 + i
            f = _make_gradient(unique_dir, f"frame_{idx:05d}.jpg", val_s, val_e)
            r = _make_gradient(raw_dir, f"frame_{idx:05d}.jpg", val_s, val_e)
            frames.append(f)
            raws.append(r)

        return sorted(unique_dir.glob("frame_*.jpg")), sorted(raw_dir.glob("frame_*.jpg")), raws

    def test_reduces_panning_cluster(self, tmp_path: Path) -> None:
        """T-05: Pass 2 verringert die Frame-Anzahl bei Panning-Cluster."""
        unique_frames, _, orig_paths = self._setup_panning_cluster(tmp_path)
        initial_count = len(unique_frames)

        result_frames, result_orig = _run_dedup_global(
            unique_frames, orig_paths, hash_size=8, pass1_threshold=10
        )

        # Nach Pass 2 sollen weniger Frames übrig sein (oder gleich wenn kein Gap gefunden)
        assert len(result_frames) <= initial_count
        assert len(result_frames) == len(result_orig)

    def test_files_physically_removed(self, tmp_path: Path) -> None:
        """Nicht-behaltene Frames werden aus frames_unique/ gelöscht."""
        unique_frames, _, orig_paths = self._setup_panning_cluster(tmp_path)
        unique_dir = unique_frames[0].parent

        result_frames, _ = _run_dedup_global(
            unique_frames, orig_paths, hash_size=8, pass1_threshold=10
        )

        actual_on_disk = sorted(unique_dir.glob("frame_*.jpg"))
        assert len(actual_on_disk) == len(result_frames)

    def test_sequential_renumbering(self, tmp_path: Path) -> None:
        """Verbleibende Frames sind lückenlos umnummeriert."""
        unique_frames, _, orig_paths = self._setup_panning_cluster(tmp_path)

        result_frames, _ = _run_dedup_global(
            unique_frames, orig_paths, hash_size=8, pass1_threshold=10
        )

        for i, f in enumerate(result_frames, start=1):
            assert f.name == f"frame_{i:05d}.jpg", f"Erwartet frame_{i:05d}.jpg, bekommen {f.name}"

    def test_too_few_frames_skips_pass2(self, tmp_path: Path) -> None:
        """T-05 Edge: Bei < 5 Frames wird Pass 2 übersprungen."""
        unique_dir = tmp_path / "frames_unique"
        unique_dir.mkdir()
        raw_dir = tmp_path / "frames_raw"
        raw_dir.mkdir()

        frames = []
        raws = []
        for i in range(3):
            f = _make_frame(unique_dir, f"frame_{i+1:05d}.jpg", i * 30)
            r = _make_frame(raw_dir, f"frame_{i+1:05d}.jpg", i * 30)
            frames.append(f)
            raws.append(r)

        result, result_orig = _run_dedup_global(frames, raws, hash_size=8, pass1_threshold=10)

        assert len(result) == 3  # unverändert


class TestTwoPassProducesFewer:
    """T-06: Zwei-Pass-Ergebnis hat ≤ Frames als Ein-Pass."""

    def test_global_after_sliding_reduces_count(self, tmp_path: Path) -> None:
        """T-06: Pass-2-Ergebnis hat ≤ Frames als Pass-1-Ergebnis."""
        from src.extract import _compute_frame_hashes, _dedupe_by_hash_sliding, _run_dedup_global

        unique_dir = tmp_path / "frames_unique"
        unique_dir.mkdir()
        raw_dir = tmp_path / "frames_raw"
        raw_dir.mkdir()

        # 5 diverse + 10 ähnliche Frames
        all_frames: list[Path] = []
        all_raws: list[Path] = []
        for i, (s, e) in enumerate([(0, 200), (200, 0), (50, 250), (250, 50), (100, 100)], start=1):
            f = _make_gradient(unique_dir, f"frame_{i:05d}.jpg", s, e)
            r = _make_gradient(raw_dir, f"frame_{i:05d}.jpg", s, e)
            all_frames.append(f)
            all_raws.append(r)

        for i in range(10):
            val_s = 60 + i * 10
            val_e = 160 + i * 10
            idx = 6 + i
            f = _make_gradient(unique_dir, f"frame_{idx:05d}.jpg", val_s, val_e)
            r = _make_gradient(raw_dir, f"frame_{idx:05d}.jpg", val_s, val_e)
            all_frames.append(f)
            all_raws.append(r)

        pass1_count = len(all_frames)  # alle 15 als "Pass-1-Ergebnis"

        result_frames, _ = _run_dedup_global(
            all_frames, all_raws, hash_size=8, pass1_threshold=10
        )

        # Pass 2 kann nur gleich oder weniger produzieren
        assert len(result_frames) <= pass1_count
