from __future__ import annotations

"""Frame-Extraktion und Deduplizierung aus YouTube-Videos."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError

from src.utils import OUTPUT_DIR, frame_to_timestamp, setup_logging, validate_dependencies


FRAMES_RAW_DIR = OUTPUT_DIR / "frames_raw"
FRAMES_UNIQUE_DIR = OUTPUT_DIR / "frames_unique"
VIDEO_FILE = OUTPUT_DIR / "video.mp4"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"


def dedupe_images(
    images: list[Path],
    threshold: int,
    hash_size: int,
) -> list[int]:
    """Findet einzigartige Bilder per Perceptual Hash und gibt deren Indizes zurück.

    Für jeden Cluster ähnlicher Bilder wird nur der erste Frame behalten.

    Args:
        images: Liste der Bildpfade in der gewünschten Sortierreihenfolge.
        threshold: Maximale Hamming-Distanz, bis zu der zwei Bilder als Duplikate gelten.
        hash_size: Größe des pHash-Gitters (NxN).

    Returns:
        Sortierte Liste der Indizes aus `images`, die behalten werden sollen.
    """
    kept_indices: list[int] = []
    kept_hashes: list[imagehash.ImageHash] = []

    for idx, path in enumerate(images):
        try:
            with Image.open(path) as img:
                h = imagehash.phash(img, hash_size=hash_size)
        except (UnidentifiedImageError, OSError):
            continue

        is_duplicate = any(abs(h - prev) <= threshold for prev in kept_hashes)
        if not is_duplicate:
            kept_indices.append(idx)
            kept_hashes.append(h)

    return kept_indices


def _download_video(url: str, max_height: int, force: bool) -> None:
    """Lädt das Video mit yt-dlp herunter.

    Args:
        url: YouTube-URL des Videos.
        max_height: Maximale vertikale Auflösung.
        force: Überschreibt existierende Datei, wenn True.
    """
    if VIDEO_FILE.exists() and not force:
        log.info("Video bereits vorhanden, überspringe Download: %s", VIDEO_FILE)
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    log.info("Lade Video herunter: %s", url)

    subprocess.run(
        [
            "yt-dlp",
            "--format",
            f"bestvideo[height<={max_height}]+bestaudio/best",
            "--output",
            str(VIDEO_FILE),
            "--no-playlist",
            url,
        ],
        check=True,
    )
    log.info("Download abgeschlossen: %s", VIDEO_FILE)


def _extract_frames(fps: float, force: bool) -> int:
    """Extrahiert Frames aus dem Video mit ffmpeg.

    Args:
        fps: Frames pro Sekunde für die Extraktion.
        force: Löscht existierende Frames und extrahiert neu, wenn True.

    Returns:
        Anzahl der extrahierten Frames.
    """
    if FRAMES_RAW_DIR.exists() and any(FRAMES_RAW_DIR.iterdir()) and not force:
        existing = sorted(FRAMES_RAW_DIR.glob("frame_*.jpg"))
        log.info(
            "Raw-Frames bereits vorhanden (%d Stück), überspringe Extraktion",
            len(existing),
        )
        return len(existing)

    FRAMES_RAW_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Extrahiere Frames mit %.1f fps ...", fps)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(VIDEO_FILE),
            "-vf",
            f"fps={fps}",
            "-q:v",
            "2",
            str(FRAMES_RAW_DIR / "frame_%05d.jpg"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    frames = sorted(FRAMES_RAW_DIR.glob("frame_*.jpg"))
    log.info("Frames extrahiert: %d", len(frames))
    return len(frames)


def _run_dedup(threshold: int, hash_size: int, force: bool) -> list[Path]:
    """Führt Perceptual-Hash-Dedup durch und kopiert Unique-Frames.

    Args:
        threshold: Hamming-Distanz-Schwellenwert.
        hash_size: pHash-Gittergröße.
        force: Leert Unique-Verzeichnis und dedupliziert neu, wenn True.

    Returns:
        Sortierte Liste der Pfade zu den unique Frames im Unique-Verzeichnis.
    """
    if FRAMES_UNIQUE_DIR.exists() and any(FRAMES_UNIQUE_DIR.iterdir()) and not force:
        unique = sorted(FRAMES_UNIQUE_DIR.glob("frame_*.jpg"))
        log.info(
            "Unique-Frames bereits vorhanden (%d Stück), überspringe Dedup",
            len(unique),
        )
        return unique

    FRAMES_UNIQUE_DIR.mkdir(parents=True, exist_ok=True)

    all_frames = sorted(FRAMES_RAW_DIR.glob("frame_*.jpg"))
    if not all_frames:
        raise RuntimeError(f"Keine Frames in {FRAMES_RAW_DIR} gefunden.")

    log.info(
        "Starte Deduplizierung: %d Frames, threshold=%d, hash_size=%d",
        len(all_frames),
        threshold,
        hash_size,
    )

    kept_indices = dedupe_images(all_frames, threshold, hash_size)

    import shutil

    for new_idx, orig_idx in enumerate(kept_indices, start=1):
        src = all_frames[orig_idx]
        dst = FRAMES_UNIQUE_DIR / f"frame_{new_idx:05d}.jpg"
        shutil.copy2(src, dst)

    unique_frames = sorted(FRAMES_UNIQUE_DIR.glob("frame_*.jpg"))
    log.info(
        "Dedup abgeschlossen: %d → %d Frames (Reduktion %.1f%%)",
        len(all_frames),
        len(unique_frames),
        (1 - len(unique_frames) / len(all_frames)) * 100 if all_frames else 0,
    )
    return unique_frames


def _write_manifest(unique_frames: list[Path], fps: float) -> None:
    """Schreibt das Manifest als JSON.

    Args:
        unique_frames: Pfade zu den beibehaltenen Frames.
        fps: Extraktions-FPS für Zeitstempel-Berechnung.
    """
    all_raw = sorted(FRAMES_RAW_DIR.glob("frame_*.jpg"))
    raw_name_to_idx = {p.name: i + 1 for i, p in enumerate(all_raw)}

    entries = []
    for frame_path in unique_frames:
        orig_num = raw_name_to_idx.get(frame_path.name, 0)
        entries.append(
            {
                "filename": frame_path.name,
                "original_frame_num": orig_num,
                "timestamp_video": frame_to_timestamp(orig_num, fps),
            }
        )

    MANIFEST_FILE.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Manifest geschrieben: %s (%d Einträge)", MANIFEST_FILE, len(entries))


def main() -> None:
    """Einstiegspunkt für die Frame-Extraktion."""
    global log
    log = setup_logging()

    parser = argparse.ArgumentParser(
        description="Extrahiert einzigartige Frames aus einem YouTube-Video."
    )
    parser.add_argument("url", help="YouTube-URL")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames pro Sekunde (Standard: 2)")
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="pHash Hamming-Distanz-Schwellenwert (Standard: 5)",
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=16,
        help="pHash-Gittergröße NxN (Standard: 16)",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=1080,
        help="Maximale Video-Höhe in Pixeln (Standard: 1080)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Existierende Daten überschreiben",
    )
    args = parser.parse_args()

    validate_dependencies()

    t_start = time.monotonic()

    _download_video(args.url, args.max_height, args.force)
    raw_count = _extract_frames(args.fps, args.force)
    unique_frames = _run_dedup(args.threshold, args.hash_size, args.force)
    _write_manifest(unique_frames, args.fps)

    elapsed = time.monotonic() - t_start
    log.info(
        "Fertig — %d raw → %d unique (%.1f%% Reduktion) in %.1fs",
        raw_count,
        len(unique_frames),
        (1 - len(unique_frames) / raw_count) * 100 if raw_count else 0,
        elapsed,
    )


log: logging.Logger

if __name__ == "__main__":
    main()
