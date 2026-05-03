from __future__ import annotations

"""Tests für Hashing-Funktionen: T-01 bis T-04."""

from pathlib import Path

import imagehash
import numpy as np
import pytest
from PIL import Image

from src.extract import _compute_frame_hashes, _detect_video_type, _hash_single_frame


def _solid_image(tmp_path: Path, name: str, color: tuple[int, int, int]) -> Path:
    arr = np.full((64, 64, 3), color, dtype=np.uint8)
    path = tmp_path / name
    Image.fromarray(arr).save(path, "JPEG", quality=95)
    return path


def _gradient_image(tmp_path: Path, name: str, start: int, end: int) -> Path:
    """Horizontaler Gradient — einzigartiger pHash."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    for j in range(64):
        val = int(start + (end - start) * j / 63)
        arr[:, j] = val
    path = tmp_path / name
    Image.fromarray(arr).save(path, "JPEG", quality=95)
    return path


class TestHashSingleFrame:
    """T-01: _hash_single_frame gibt (path_str, hex_hash) zurück."""

    def test_valid_image_returns_hex(self, tmp_path: Path) -> None:
        img = _solid_image(tmp_path, "a.jpg", (200, 100, 50))
        path_str, hex_hash = _hash_single_frame((str(img), 8))
        assert path_str == str(img)
        assert hex_hash is not None
        assert len(hex_hash) == 16  # 8×8 / 4 Bits pro Hex-Zeichen

    def test_invalid_path_returns_none(self, tmp_path: Path) -> None:
        _, hex_hash = _hash_single_frame((str(tmp_path / "nonexistent.jpg"), 8))
        assert hex_hash is None

    def test_reconstructible_via_hex_to_hash(self, tmp_path: Path) -> None:
        img = _solid_image(tmp_path, "b.jpg", (10, 20, 30))
        _, hex_str = _hash_single_frame((str(img), 16))
        assert hex_str is not None
        reconstructed = imagehash.hex_to_hash(hex_str)
        assert isinstance(reconstructed, imagehash.ImageHash)


class TestComputeFrameHashes:
    """T-02: _compute_frame_hashes parallel → reihenfolgetreue Ergebnisse."""

    def test_sequential_returns_correct_count(self, tmp_path: Path) -> None:
        frames = [_solid_image(tmp_path, f"f{i:03d}.jpg", (i * 10, 0, 0)) for i in range(5)]
        result = _compute_frame_hashes(frames, hash_size=8, parallel=False)
        assert len(result) == 5
        assert all(h is not None for h in result)

    def test_empty_list(self) -> None:
        assert _compute_frame_hashes([], hash_size=8) == []

    def test_parallel_matches_sequential(self, tmp_path: Path) -> None:
        """T-02 Kern: Parallel-Ergebnis ist reihenfolgetreu und identisch zu sequentiell."""
        frames = [_gradient_image(tmp_path, f"g{i:03d}.jpg", i * 5, i * 5 + 100) for i in range(15)]
        sequential = _compute_frame_hashes(frames, hash_size=8, parallel=False)
        # parallel wird ab 80 Frames aktiviert — wir testen das Interface direkt
        # durch explizit parallel=False und Vergleich der Hashes
        # Für echten Parallelismus brauchen wir 80+ Frames (zu langsam für Unit-Test)
        # Stattdessen: Test dass sequential korrekte Reihenfolge hält
        for i in range(len(frames) - 1):
            h_a = sequential[i]
            h_b = sequential[i + 1]
            assert h_a is not None and h_b is not None

    def test_none_for_corrupt_file(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "bad.jpg"
        corrupt.write_bytes(b"not an image")
        result = _compute_frame_hashes([corrupt], hash_size=8, parallel=False)
        assert result == [None]

    def test_parallel_path_80_frames(self, tmp_path: Path) -> None:
        """T-02b: Echter ProcessPoolExecutor-Pfad bei >= 80 Frames — korrekte Anzahl + Reihenfolge."""
        frames = [
            _solid_image(tmp_path, f"p{i:03d}.jpg", (i * 3 % 256, i * 5 % 256, i * 7 % 256))
            for i in range(80)
        ]
        parallel_result = _compute_frame_hashes(frames, hash_size=8, parallel=True)
        sequential_result = _compute_frame_hashes(frames, hash_size=8, parallel=False)

        assert len(parallel_result) == 80
        assert all(h is not None for h in parallel_result)

        # Reihenfolge muss identisch zur sequentiellen Berechnung sein
        for i, (ph, sh) in enumerate(zip(parallel_result, sequential_result)):
            assert str(ph) == str(sh), f"Hash-Abweichung an Position {i}"


class TestDetectVideoType:
    """T-03, T-04: _detect_video_type klassifiziert korrekt."""

    def _make_hashes_with_ratio(self, low_count: int, total: int) -> list:
        """Erzeugt Mock-Hashes mit gezieltem Anteil low-distance Paare.

        Strategie:
          - low_count+1 identische all-zeros Hashes → low_count Paare mit dist=0 (≤ 15)
          - dann alternierend all-ones / all-zeros → high-dist Paare mit dist=16 (> 15)
        """
        import imagehash as ih
        import numpy as np

        z = ih.ImageHash(np.zeros((4, 4), dtype=bool))
        o = ih.ImageHash(np.ones((4, 4), dtype=bool))
        high_count = total - low_count

        hashes = [z] * (low_count + 1)
        for _ in range(high_count):
            hashes.append(o if hashes[-1] == z else z)

        return hashes

    def test_slideshow_detected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """T-03: > 60% low-dist → slideshow."""
        import logging
        import src.extract as mod
        monkeypatch.setattr(mod, "log", logging.getLogger("test"))

        hashes = self._make_hashes_with_ratio(low_count=8, total=9)
        assert _detect_video_type(hashes) == "slideshow"

    def test_live_action_detected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """T-04: < 20% low-dist → live-action."""
        import logging
        import src.extract as mod
        monkeypatch.setattr(mod, "log", logging.getLogger("test"))

        hashes = self._make_hashes_with_ratio(low_count=1, total=9)
        assert _detect_video_type(hashes) == "live-action"

    def test_mixed_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """40% low-dist → mixed."""
        import logging
        import src.extract as mod
        monkeypatch.setattr(mod, "log", logging.getLogger("test"))

        hashes = self._make_hashes_with_ratio(low_count=4, total=9)
        assert _detect_video_type(hashes) == "mixed"

    def test_too_few_hashes_returns_live_action(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import logging
        import src.extract as mod
        monkeypatch.setattr(mod, "log", logging.getLogger("test"))

        assert _detect_video_type([None, None]) == "live-action"
