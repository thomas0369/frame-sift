from __future__ import annotations

"""Frame-Extraktion und Deduplizierung aus YouTube-Videos."""

import argparse
import json
import logging
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, urlparse

import imagehash
from PIL import Image, UnidentifiedImageError

from src.utils import OUTPUT_DIR, frame_to_timestamp, setup_logging, validate_dependencies


def _extract_video_id(url: str) -> str:
    """Extrahiert die YouTube-Video-ID aus einer URL.

    Unterstützt youtube.com/watch?v=ID, youtu.be/ID und youtube.com/shorts/ID.

    Raises:
        ValueError: Wenn keine ID gefunden werden kann.
    """
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/").split("?")[0]
    qs = parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    parts = [p for p in parsed.path.split("/") if p]
    if parts:
        return parts[-1]
    raise ValueError(f"Keine Video-ID in URL gefunden: {url}")


# ---------------------------------------------------------------------------
# Phase B: Pool-Worker (top-level, picklable)
# ---------------------------------------------------------------------------

def _hash_single_frame(args: tuple[str, int]) -> tuple[str, str | None]:
    """Pool-Worker: öffnet Bild im Child-Prozess und gibt (path_str, hex_hash) zurück."""
    path_str, hash_size = args
    try:
        from PIL import Image  # lokaler Import — child-Prozess hat eigenen Namespace
        import imagehash as _ih
        with Image.open(path_str) as img:
            return path_str, str(_ih.phash(img, hash_size=hash_size))
    except Exception:
        return path_str, None


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _compute_frame_hashes(
    frames: list[Path],
    hash_size: int,
    parallel: bool = True,
) -> list[imagehash.ImageHash | None]:
    """Berechnet pHash für alle Frames. Parallelisiert bei ≥ 80 Frames."""
    if not frames:
        return []

    if parallel and len(frames) >= 80:
        log.info("Starte paralleles Hashing: %d Frames ...", len(frames))
        try:
            args_list = [(str(f), hash_size) for f in frames]
            with ProcessPoolExecutor() as pool:
                raw = list(pool.map(_hash_single_frame, args_list))
            path_to_hex: dict[str, str | None] = {p: h for p, h in raw}
            return [
                imagehash.hex_to_hash(path_to_hex[str(f)])
                if path_to_hex.get(str(f))
                else None
                for f in frames
            ]
        except Exception as exc:
            log.warning("Paralleles Hashing fehlgeschlagen (%s) — falle auf sequentiell zurück", exc)

    log.info("Berechne pHash für %d Frames (sequentiell) ...", len(frames))
    result: list[imagehash.ImageHash | None] = []
    for path in frames:
        try:
            with Image.open(path) as img:
                result.append(imagehash.phash(img, hash_size=hash_size))
        except (UnidentifiedImageError, OSError):
            result.append(None)
    return result


# ---------------------------------------------------------------------------
# Phase A: Video-Typ-Erkennung + Adaptive FPS
# ---------------------------------------------------------------------------

def _presample_for_type(
    video_file: Path,
    n_frames: int = 10,
    duration: int = 30,
) -> list[imagehash.ImageHash | None]:
    """Extrahiert kurze Frame-Sequenz aus dem Video-Anfang für Typ-Erkennung.

    Läuft in einem temporären Verzeichnis, hinterlässt keine persistenten Dateien.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_file),
                "-t", str(duration),
                "-vf", "fps=1",
                "-q:v", "5",
                f"{tmpdir}/frame_%05d.jpg",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        sample = sorted(Path(tmpdir).glob("frame_*.jpg"))[:n_frames]
        return _compute_frame_hashes(sample, hash_size=8, parallel=False)


def _detect_video_type(
    hashes: list[imagehash.ImageHash | None],
) -> Literal["slideshow", "mixed", "live-action"]:
    """Klassifiziert den Video-Typ anhand des Low-Distanz-Anteils aufeinanderfolgender Frames.

    Schwellenwerte:
      > 60 % low-dist (≤ 15) → slideshow  → 1 fps empfohlen
      < 20 % low-dist        → live-action → 2 fps
      dazwischen             → mixed       → 2 fps (konservativ)
    """
    valid = [h for h in hashes if h is not None]
    if len(valid) < 3:
        return "live-action"
    dists = [abs(valid[i] - valid[i + 1]) for i in range(len(valid) - 1)]
    low_count = sum(1 for d in dists if d <= 15)
    ratio = low_count / len(dists)

    if ratio > 0.60:
        video_type: Literal["slideshow", "mixed", "live-action"] = "slideshow"
    elif ratio < 0.20:
        video_type = "live-action"
    else:
        video_type = "mixed"

    log.info("Video-Typ erkannt: %s (%.0f%% low-dist Paare)", video_type, ratio * 100)
    return video_type


# ---------------------------------------------------------------------------
# Threshold-Erkennung
# ---------------------------------------------------------------------------

def _detect_threshold(hashes: list[imagehash.ImageHash | None]) -> int:
    """Findet automatisch den optimalen Deduplizierungs-Schwellenwert via Valley Detection.

    Analysiert die Verteilung aufeinanderfolgender pHash-Abstände. Videos mit
    Ken-Burns-Effekt erzeugen eine bimodale Verteilung: ein dichter linker Cluster
    (Pseudo-Duplikate, Zoom-Varianten) und ein rechter Cluster (echte Szenenwechsel).
    Der Threshold wird am Tal zwischen den beiden Clustern gesetzt.

    Returns:
        Erkannter Threshold oder 10 als Fallback.
    """
    valid = [h for h in hashes if h is not None]
    if len(valid) < 4:
        log.warning("Zu wenige Frames für Auto-Threshold — verwende Fallback 10")
        return 10

    dists = [abs(valid[i] - valid[i + 1]) for i in range(len(valid) - 1)]
    max_dist = max(dists)
    n_buckets = max(max_dist // 10 + 2, 3)
    buckets = [0] * n_buckets
    for d in dists:
        buckets[d // 10] += 1

    peak_count = max(buckets)
    cutoff = max(peak_count * 0.10, 2)

    log.debug("pHash-Distanz-Histogramm (Raw-Frames, peak=%d, cutoff=%.1f):", peak_count, cutoff)
    for i, count in enumerate(buckets):
        if i * 10 > max_dist + 10:
            break
        bar = "#" * min(count, 40)
        log.debug("  %3d-%3d: %3d %s", i * 10, i * 10 + 9, count, bar)

    for i in range(len(buckets) - 1):
        if buckets[i] <= cutoff and buckets[i + 1] <= cutoff:
            detected = i * 10
            scene_changes = sum(buckets[i:])
            log.info(
                "Auto-Threshold erkannt: %d  (Tal ab Bucket %d, ~%d echte Szenenwechsel)",
                detected,
                i * 10,
                scene_changes,
            )
            return detected

    log.warning("Kein eindeutiges Tal gefunden — verwende Fallback threshold=10")
    return 10


def _detect_pass2_threshold(
    hashes: list[imagehash.ImageHash | None],
    pass1_threshold: int,
) -> int | None:
    """Findet den Pass-2-Threshold durch Gap-Analyse der sortierten Unique-Frame-Distanzen.

    Sucht den größten Sprung in den sortierten consecutiven Abständen der Unique-Frames.
    Der Gap zwischen dem Panning-Cluster (dist=22–36) und echten Schnitten (dist=50+)
    ergibt einen robusten Threshold für den zweiten Dedup-Pass.

    Returns:
        Erkannter Threshold oder None wenn kein sinnvoller Gap gefunden.
    """
    valid = [h for h in hashes if h is not None]
    if len(valid) < 4:
        return None

    dists = [abs(valid[i] - valid[i + 1]) for i in range(len(valid) - 1)]
    relevant = sorted(d for d in dists if d > pass1_threshold)

    if len(relevant) < 3:
        return None

    max_gap = 0
    gap_pos = 0
    for i in range(len(relevant) - 1):
        gap = relevant[i + 1] - relevant[i]
        if gap > max_gap:
            max_gap = gap
            gap_pos = i

    if max_gap < 8:
        log.info("Pass 2: kein sinnvoller Gap (max=%d) — überspringe zweiten Pass", max_gap)
        return None

    threshold = (relevant[gap_pos] + relevant[gap_pos + 1]) // 2
    log.info(
        "Pass-2-Threshold erkannt: %d  (Gap %d→%d, Spanne %d)",
        threshold,
        relevant[gap_pos],
        relevant[gap_pos + 1],
        max_gap,
    )
    return threshold


# ---------------------------------------------------------------------------
# Deduplizierung
# ---------------------------------------------------------------------------

def _dedupe_by_hash(
    hashes: list[imagehash.ImageHash | None],
    threshold: int,
) -> list[int]:
    """Globaler Modus: Frame wird behalten wenn er zu KEINEM bereits gehaltenen Frame ähnelt."""
    kept_indices: list[int] = []
    kept_hashes: list[imagehash.ImageHash] = []

    for idx, h in enumerate(hashes):
        if h is None:
            continue
        is_duplicate = any(abs(h - prev) <= threshold for prev in kept_hashes)
        if not is_duplicate:
            kept_indices.append(idx)
            kept_hashes.append(h)

    return kept_indices


def _dedupe_by_hash_sliding(
    hashes: list[imagehash.ImageHash | None],
    threshold: int,
) -> list[int]:
    """Sliding-Window-Modus: Frame wird nur gegen den unmittelbar vorherigen Raw-Frame verglichen.

    Misst die momentane Änderung statt der kumulativen Drift. Ideal für Ken-Burns-Sequenzen:
    jede kontinuierliche Zoom-Sequenz zählt als ein Duplikat, harte Schnitte werden immer erfasst.
    """
    kept_indices: list[int] = []

    for idx, h in enumerate(hashes):
        if h is None:
            continue
        prev = hashes[idx - 1] if idx > 0 else None
        if prev is None and idx > 0:
            for j in range(idx - 1, -1, -1):
                if hashes[j] is not None:
                    prev = hashes[j]
                    break
        if prev is None or abs(h - prev) > threshold:
            kept_indices.append(idx)

    return kept_indices


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
    hashes = _compute_frame_hashes(images, hash_size)
    return _dedupe_by_hash(hashes, threshold)


# ---------------------------------------------------------------------------
# Phase C: Zweistufiges Dedup (Pass 2)
# ---------------------------------------------------------------------------

def _run_dedup_global(
    unique_frames: list[Path],
    orig_paths: list[Path],
    hash_size: int,
    pass1_threshold: int,
) -> tuple[list[Path], list[Path]]:
    """Pass 2: Globaler Dedup auf den Sliding-gefilterten Unique-Frames.

    Erkennt den Threshold automatisch per Gap-Analyse, löscht nicht-behaltene Frames
    physisch aus frames_unique/ und nummeriert die Verbleibenden neu.

    Args:
        unique_frames: Pfade in frames_unique/ nach Pass 1.
        orig_paths:    Korrespondierende Raw-Frame-Pfade.
        hash_size:     pHash-Gittergröße.
        pass1_threshold: Threshold aus Pass 1 (untere Grenze für Gap-Suche).

    Returns:
        (gefilterte_unique_frames, gefilterte_orig_paths)
    """
    if len(unique_frames) < 5:
        log.info("Pass 2: zu wenige Frames (%d) — überspringe", len(unique_frames))
        return unique_frames, orig_paths

    hashes = _compute_frame_hashes(unique_frames, hash_size, parallel=False)
    pass2_threshold = _detect_pass2_threshold(hashes, pass1_threshold)

    if pass2_threshold is None:
        return unique_frames, orig_paths

    kept_indices = _dedupe_by_hash(hashes, pass2_threshold)

    if len(kept_indices) < 5:
        log.warning(
            "Pass 2 würde auf %d Frames reduzieren (< 5) — überspringe",
            len(kept_indices),
        )
        return unique_frames, orig_paths

    kept_set = set(kept_indices)
    for i, f in enumerate(unique_frames):
        if i not in kept_set:
            f.unlink()

    frames_unique_dir = unique_frames[0].parent
    remaining = sorted(frames_unique_dir.glob("frame_*.jpg"))
    new_paths: list[Path] = []
    for new_idx, old_path in enumerate(remaining, start=1):
        new_path = frames_unique_dir / f"frame_{new_idx:05d}.jpg"
        if old_path != new_path:
            old_path.rename(new_path)
        new_paths.append(new_path)

    kept_orig = [orig_paths[i] for i in kept_indices]
    log.info(
        "Pass 2 (global): %d → %d Frames (threshold=%d)",
        len(unique_frames),
        len(new_paths),
        pass2_threshold,
    )
    return new_paths, kept_orig


# ---------------------------------------------------------------------------
# Download + Extraktion
# ---------------------------------------------------------------------------

def _download_video(url: str, project_dir: Path, max_height: int, force: bool) -> None:
    """Lädt das Video mit yt-dlp herunter."""
    video_file = project_dir / "video.mp4"
    if video_file.exists() and not force:
        log.info("Video bereits vorhanden, überspringe Download: %s", video_file)
        return

    project_dir.mkdir(parents=True, exist_ok=True)
    log.info("Lade Video herunter: %s", url)

    subprocess.run(
        [
            "yt-dlp",
            "--format",
            f"bestvideo[height<={max_height}]+bestaudio/best",
            "--merge-output-format",
            "mp4",
            "--output",
            str(video_file),
            "--no-playlist",
            url,
        ],
        check=True,
    )
    log.info("Download abgeschlossen: %s", video_file)


def _read_fps_from_manifest(project_dir: Path) -> float | None:
    """Liest fps aus dem bestehenden Manifest, falls vorhanden."""
    manifest = project_dir / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data.get("meta", {}).get("fps")
        except Exception:
            pass
    return None


def _extract_frames(project_dir: Path, fps: float, force: bool) -> int:
    """Extrahiert Frames aus dem Video mit ffmpeg."""
    frames_raw_dir = project_dir / "frames_raw"
    video_file = project_dir / "video.mp4"

    if frames_raw_dir.exists() and any(frames_raw_dir.iterdir()) and not force:
        existing = sorted(frames_raw_dir.glob("frame_*.jpg"))
        log.info(
            "Raw-Frames bereits vorhanden (%d Stück), überspringe Extraktion",
            len(existing),
        )
        return len(existing)

    import shutil as _shutil
    if frames_raw_dir.exists():
        _shutil.rmtree(frames_raw_dir)
    frames_raw_dir.mkdir(parents=True, exist_ok=True)

    log.info("Extrahiere Frames mit %.1f fps ...", fps)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_file),
            "-vf",
            f"fps={fps}",
            "-q:v",
            "2",
            str(frames_raw_dir / "frame_%05d.jpg"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    frames = sorted(frames_raw_dir.glob("frame_*.jpg"))
    log.info("Frames extrahiert: %d", len(frames))
    return len(frames)


def _run_dedup(
    project_dir: Path,
    threshold: int,
    hash_size: int,
    force: bool,
    dedup_mode: str = "sliding",
    precomputed_hashes: list[imagehash.ImageHash | None] | None = None,
) -> tuple[list[Path], list[Path] | None]:
    """Führt Perceptual-Hash-Dedup (Pass 1) durch und kopiert Unique-Frames.

    Returns:
        (unique_frames, orig_raw_paths) — orig_raw_paths ist None wenn aus Cache geladen.
    """
    frames_raw_dir = project_dir / "frames_raw"
    frames_unique_dir = project_dir / "frames_unique"

    if frames_unique_dir.exists() and any(frames_unique_dir.iterdir()) and not force:
        unique = sorted(frames_unique_dir.glob("frame_*.jpg"))
        log.info(
            "Unique-Frames bereits vorhanden (%d Stück), überspringe Dedup",
            len(unique),
        )
        return unique, None

    import shutil as _shutil
    if frames_unique_dir.exists():
        _shutil.rmtree(frames_unique_dir)
    frames_unique_dir.mkdir(parents=True, exist_ok=True)

    all_frames = sorted(frames_raw_dir.glob("frame_*.jpg"))
    if not all_frames:
        raise RuntimeError(f"Keine Frames in {frames_raw_dir} gefunden.")

    hashes = (
        precomputed_hashes
        if precomputed_hashes is not None
        else _compute_frame_hashes(all_frames, hash_size)
    )

    log.info(
        "Starte Deduplizierung (Pass 1): %d Frames, threshold=%d, modus=%s",
        len(all_frames),
        threshold,
        dedup_mode,
    )

    if dedup_mode == "sliding":
        kept_indices = _dedupe_by_hash_sliding(hashes, threshold)
    else:
        kept_indices = _dedupe_by_hash(hashes, threshold)

    orig_raw_paths: list[Path] = []
    for new_idx, orig_idx in enumerate(kept_indices, start=1):
        src = all_frames[orig_idx]
        dst = frames_unique_dir / f"frame_{new_idx:05d}.jpg"
        _shutil.copy2(src, dst)
        orig_raw_paths.append(src)

    unique_frames = sorted(frames_unique_dir.glob("frame_*.jpg"))
    log.info(
        "Pass 1 abgeschlossen: %d → %d Frames (Reduktion %.1f%%)",
        len(all_frames),
        len(unique_frames),
        (1 - len(unique_frames) / len(all_frames)) * 100 if all_frames else 0,
    )
    return unique_frames, orig_raw_paths


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _write_manifest(
    project_dir: Path,
    unique_frames: list[Path],
    orig_raw_paths: list[Path],
    fps: float,
    threshold_used: int,
    auto_threshold: bool,
    dedup_mode: str = "sliding",
    pass1_count: int | None = None,
    pass2_count: int | None = None,
    video_type: str | None = None,
) -> None:
    """Schreibt das Manifest als JSON mit Meta-Block und Frame-Liste."""
    frames_raw_dir = project_dir / "frames_raw"
    manifest_file = project_dir / "manifest.json"

    all_raw = sorted(frames_raw_dir.glob("frame_*.jpg"))

    frame_entries = []
    for frame_path, raw_path in zip(unique_frames, orig_raw_paths):
        orig_num = int(raw_path.stem.split("_")[1])
        frame_entries.append(
            {
                "filename": frame_path.name,
                "original_frame_num": orig_num,
                "timestamp_video": frame_to_timestamp(orig_num, fps),
            }
        )

    meta: dict = {
        "threshold_used": threshold_used,
        "auto_threshold": auto_threshold,
        "dedup_mode": dedup_mode,
        "hash_size": 16,
        "fps": fps,
        "raw_frame_count": len(all_raw),
        "unique_frame_count": len(unique_frames),
    }
    if video_type is not None:
        meta["video_type"] = video_type
    if pass1_count is not None:
        meta["pass1_count"] = pass1_count
    if pass2_count is not None:
        meta["pass2_count"] = pass2_count

    manifest = {"meta": meta, "frames": frame_entries}
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Manifest geschrieben: %s (%d Einträge)", manifest_file, len(frame_entries))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Einstiegspunkt für die Frame-Extraktion."""
    global log

    parser = argparse.ArgumentParser(
        description="Extrahiert einzigartige Frames aus einem YouTube-Video."
    )
    parser.add_argument("url", help="YouTube-URL")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frames pro Sekunde (Standard: automatisch — 1 fps für Slideshows, 2 fps sonst)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="pHash Hamming-Distanz-Schwellenwert (Standard: automatisch erkannt)",
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
        "--dedup-mode",
        choices=["sliding", "global"],
        default="sliding",
        help=(
            "sliding: vergleicht jeden Frame nur mit dem unmittelbar vorherigen (Standard) | "
            "global: vergleicht gegen alle bisher gehaltenen Frames"
        ),
    )
    parser.add_argument(
        "--no-second-pass",
        action="store_true",
        help="Zweiten globalen Dedup-Pass deaktivieren (nur bei --dedup-mode sliding relevant)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Existierende Daten überschreiben",
    )
    args = parser.parse_args()

    video_id = _extract_video_id(args.url)
    project_dir = OUTPUT_DIR / video_id
    project_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(log_dir=project_dir)
    log.info("Projekt-Verzeichnis: %s", project_dir)

    validate_dependencies()

    t_start = time.monotonic()

    _download_video(args.url, project_dir, args.max_height, args.force)

    # FPS bestimmen: explizit → Manifest-Cache → Video-Typ-Erkennung
    frames_raw_dir = project_dir / "frames_raw"
    video_type: str | None = None

    if args.fps is not None:
        fps = args.fps
    elif frames_raw_dir.exists() and any(frames_raw_dir.iterdir()) and not args.force:
        fps = _read_fps_from_manifest(project_dir) or 2.0
        log.info("FPS aus Manifest übernommen: %.1f", fps)
    else:
        log.info("Analysiere Video-Typ (Pre-Sample) ...")
        sample_hashes = _presample_for_type(project_dir / "video.mp4")
        video_type = _detect_video_type(sample_hashes)
        fps = 1.0 if video_type == "slideshow" else 2.0
        log.info("Adaptive FPS: %.1f (Video-Typ: %s)", fps, video_type)

    raw_count = _extract_frames(project_dir, fps, args.force)

    all_raw = sorted((project_dir / "frames_raw").glob("frame_*.jpg"))
    hashes = _compute_frame_hashes(all_raw, args.hash_size)

    auto_threshold = args.threshold is None
    if auto_threshold:
        threshold_used = _detect_threshold(hashes)
    else:
        threshold_used = args.threshold
        log.info("Verwende manuellen Threshold: %d", threshold_used)

    unique_frames, orig_raw_paths = _run_dedup(
        project_dir,
        threshold_used,
        args.hash_size,
        args.force,
        dedup_mode=args.dedup_mode,
        precomputed_hashes=hashes,
    )

    pass1_count: int | None = None
    pass2_count: int | None = None

    if orig_raw_paths is not None:
        pass1_count = len(unique_frames)

        if args.dedup_mode == "sliding" and not args.no_second_pass:
            unique_frames, orig_raw_paths = _run_dedup_global(
                unique_frames, orig_raw_paths, args.hash_size, threshold_used
            )
            pass2_count = len(unique_frames)

        _write_manifest(
            project_dir,
            unique_frames,
            orig_raw_paths,
            fps,
            threshold_used,
            auto_threshold,
            args.dedup_mode,
            pass1_count=pass1_count,
            pass2_count=pass2_count,
            video_type=video_type,
        )

    elapsed = time.monotonic() - t_start
    log.info(
        "Fertig — %d raw → %d unique (%.1f%% Reduktion) in %.1fs  [threshold=%d%s, fps=%.1f]",
        raw_count,
        len(unique_frames),
        (1 - len(unique_frames) / raw_count) * 100 if raw_count else 0,
        elapsed,
        threshold_used,
        " auto" if auto_threshold else "",
        fps,
    )


log: logging.Logger = logging.getLogger("frame_sift")

if __name__ == "__main__":
    main()
