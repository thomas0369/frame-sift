from __future__ import annotations

"""Prompt-Generator für die Claude-Code-Vision-Analyse."""

import argparse
import json
import sys
from pathlib import Path

from src.utils import OUTPUT_DIR, setup_logging


FRAMES_UNIQUE_DIR = OUTPUT_DIR / "frames_unique"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"
ANALYSIS_PROMPT_FILE = OUTPUT_DIR / "ANALYSIS_PROMPT.md"

PROMPT_HEADER = """\
# Frame-Analyse-Auftrag

Analysiere die Bilder in `output/frames_unique/`. Pro Bild liefere als JSON-Eintrag:

- `filename`
- `timestamp` (aus manifest.json)
- `texts` (alle wörtlich erkennbaren Texte als Liste)
- `symbols` (Symbole, Logos, Icons)
- `numbers` (alle sichtbaren Zahlen)
- `people` (erkennbare Personen)
- `places` (erkennbare Orte)
- `notable` (Auffälligkeiten)

Schreibe Ergebnis als JSON-Array nach `output/analysis.json` UND als Markdown-Tabelle \
nach `output/analysis.md`. Append-Modus zwischen Batches, nicht überschreiben.

"""


def _load_manifest() -> dict[str, str]:
    """Liest das Manifest und gibt ein Dict {filename → timestamp} zurück."""
    if not MANIFEST_FILE.exists():
        return {}
    entries = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return {e["filename"]: e["timestamp_video"] for e in entries}


def _get_frame_paths() -> list[Path]:
    """Gibt alle unique Frames sortiert zurück.

    Raises:
        SystemExit: Wenn das Unique-Frames-Verzeichnis leer oder nicht vorhanden ist.
    """
    if not FRAMES_UNIQUE_DIR.exists():
        print(
            f"Fehler: {FRAMES_UNIQUE_DIR} existiert nicht. "
            "Bitte zuerst 'python -m src.extract' ausführen.",
            file=sys.stderr,
        )
        sys.exit(1)

    frames = sorted(FRAMES_UNIQUE_DIR.glob("frame_*.jpg"))
    if not frames:
        print(
            f"Fehler: Keine Frames in {FRAMES_UNIQUE_DIR} gefunden.",
            file=sys.stderr,
        )
        sys.exit(1)

    return frames


def _build_batches(frames: list[Path], batch_size: int) -> list[list[Path]]:
    return [frames[i : i + batch_size] for i in range(0, len(frames), batch_size)]


def mode_list(frames: list[Path], batch_size: int) -> None:
    """Gibt Frame-Pfade sortiert und in Batches gruppiert auf stdout aus.

    Args:
        frames: Alle unique Frame-Pfade.
        batch_size: Anzahl Frames pro Batch.
    """
    batches = _build_batches(frames, batch_size)
    for batch_num, batch in enumerate(batches, start=1):
        print(f"--- Batch {batch_num} ---")
        for path in batch:
            print(str(path))
    print(f"\nGesamt: {len(frames)} Frames in {len(batches)} Batches")


def mode_prompt(frames: list[Path], batch_size: int) -> None:
    """Generiert output/ANALYSIS_PROMPT.md mit Batch-Listen.

    Args:
        frames: Alle unique Frame-Pfade.
        batch_size: Anzahl Frames pro Batch.
    """
    timestamps = _load_manifest()
    batches = _build_batches(frames, batch_size)

    lines: list[str] = [PROMPT_HEADER]
    for batch_num, batch in enumerate(batches, start=1):
        lines.append(f"## Batch {batch_num}\n")
        for path in batch:
            ts = timestamps.get(path.name, "")
            comment = f"  <!-- {ts} -->" if ts else ""
            lines.append(f"- {path}{comment}")
        lines.append("")

    ANALYSIS_PROMPT_FILE.write_text("\n".join(lines), encoding="utf-8")
    log.info(
        "Prompt geschrieben: %s (%d Frames, %d Batches)",
        ANALYSIS_PROMPT_FILE,
        len(frames),
        len(batches),
    )
    print(f"Prompt gespeichert: {ANALYSIS_PROMPT_FILE}")


def main() -> None:
    """Einstiegspunkt für den Prompt-Generator."""
    global log
    log = setup_logging()

    parser = argparse.ArgumentParser(
        description="Generiert Frame-Listen und Analyse-Prompts für Claude Code."
    )
    parser.add_argument(
        "--mode",
        choices=["list", "prompt"],
        default="list",
        help="list: Pfade auf stdout | prompt: ANALYSIS_PROMPT.md erzeugen (Standard: list)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Frames pro Batch (Standard: 20)",
    )
    args = parser.parse_args()

    frames = _get_frame_paths()
    log.info("Gefunden: %d unique Frames", len(frames))

    if args.mode == "list":
        mode_list(frames, args.batch_size)
    else:
        mode_prompt(frames, args.batch_size)


log: logging.Logger

if __name__ == "__main__":
    main()
