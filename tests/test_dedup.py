from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.extract import dedupe_images


def _make_checkerboard(tmp_path: Path, name: str, tile: int = 20) -> Path:
    """Schachbrettmuster — komplexe Frequenzstruktur, einzigartiger pHash."""
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        for j in range(200):
            if (i // tile + j // tile) % 2 == 0:
                arr[i, j] = [255, 255, 255]
    path = tmp_path / name
    Image.fromarray(arr).save(path, "JPEG", quality=95)
    return path


def _make_hstripes(tmp_path: Path, name: str, stripe: int = 20) -> Path:
    """Horizontale Streifen — klar anderer pHash als Schachbrett."""
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        if (i // stripe) % 2 == 0:
            arr[i, :] = [255, 255, 255]
    path = tmp_path / name
    Image.fromarray(arr).save(path, "JPEG", quality=95)
    return path


def _make_vstripes(tmp_path: Path, name: str, stripe: int = 20) -> Path:
    """Vertikale Streifen — klar anderer pHash als horizontale Streifen."""
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    for j in range(200):
        if (j // stripe) % 2 == 0:
            arr[:, j] = [255, 255, 255]
    path = tmp_path / name
    Image.fromarray(arr).save(path, "JPEG", quality=95)
    return path


class TestDedupeImages:
    def test_zwei_identische_ein_unikat(self, tmp_path: Path) -> None:
        """Zwei nahezu identische Bilder + ein abweichendes → zwei Cluster.

        hstripes_20px vs hstripes_22px: Hamming-Distanz ~7 (< threshold=8 → Duplikate)
        hstripes vs checkerboard: Hamming-Distanz ~39 (> threshold=8 → verschieden)
        """
        img_a = _make_hstripes(tmp_path, "a.jpg", stripe=20)
        img_b = _make_hstripes(tmp_path, "b.jpg", stripe=22)
        img_c = _make_checkerboard(tmp_path, "c.jpg")

        result = dedupe_images([img_a, img_b, img_c], threshold=8, hash_size=16)

        assert len(result) == 2, f"Erwartet 2 Unikate, erhalten: {result}"
        assert 0 in result, "Erster Frame muss behalten werden"
        assert 2 in result, "Klar abweichendes Bild muss behalten werden"

    def test_alle_verschieden(self, tmp_path: Path) -> None:
        """Drei strukturell verschiedene Bilder → drei Unikate.

        Alle paarweisen Hamming-Distanzen > 5 (threshold).
        """
        img_a = _make_checkerboard(tmp_path, "a.jpg")
        img_b = _make_hstripes(tmp_path, "b.jpg")
        img_c = _make_vstripes(tmp_path, "c.jpg")

        result = dedupe_images([img_a, img_b, img_c], threshold=5, hash_size=16)

        assert len(result) == 3

    def test_erster_frame_wird_behalten(self, tmp_path: Path) -> None:
        """Bei Duplikaten bleibt stets der erste Frame (Index 0).

        Identischer Numpy-Array → pHash-Distanz 0.
        """
        img_a = _make_checkerboard(tmp_path, "a.jpg")
        img_b = _make_checkerboard(tmp_path, "b.jpg")

        result = dedupe_images([img_a, img_b], threshold=5, hash_size=16)

        assert result == [0]

    def test_leere_liste(self, tmp_path: Path) -> None:
        result = dedupe_images([], threshold=5, hash_size=16)
        assert result == []
