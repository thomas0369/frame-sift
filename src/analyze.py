from __future__ import annotations

"""Prompt-Generator für die Claude-Code-Vision-Analyse."""

import argparse
import json
import sys
from pathlib import Path

from src.utils import OUTPUT_DIR, setup_logging


def _load_manifest(manifest_file: Path) -> dict[str, str]:
    """Liest das Manifest und gibt ein Dict {filename → timestamp} zurück.

    Unterstützt sowohl das alte Format (JSON-Array) als auch das neue Format
    (JSON-Objekt mit "meta" und "frames"-Schlüsseln).
    """
    if not manifest_file.exists():
        return {}
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    entries = data["frames"] if isinstance(data, dict) else data
    return {e["filename"]: e["timestamp_video"] for e in entries}


def _get_frame_paths(frames_unique_dir: Path) -> list[Path]:
    """Gibt alle unique Frames sortiert zurück.

    Raises:
        SystemExit: Wenn das Unique-Frames-Verzeichnis leer oder nicht vorhanden ist.
    """
    if not frames_unique_dir.exists():
        print(
            f"Fehler: {frames_unique_dir} existiert nicht. "
            "Bitte zuerst 'python -m src.extract <URL>' ausführen.",
            file=sys.stderr,
        )
        sys.exit(1)

    frames = sorted(frames_unique_dir.glob("frame_*.jpg"))
    if not frames:
        print(
            f"Fehler: Keine Frames in {frames_unique_dir} gefunden.",
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


def mode_prompt(
    frames: list[Path],
    batch_size: int,
    manifest_file: Path,
    analysis_prompt_file: Path,
    project_dir: Path,
) -> None:
    """Generiert ANALYSIS_PROMPT.md im Projektverzeichnis mit Batch-Listen.

    Args:
        frames: Alle unique Frame-Pfade.
        batch_size: Anzahl Frames pro Batch.
        manifest_file: Pfad zur manifest.json.
        analysis_prompt_file: Ausgabepfad für den Prompt.
        project_dir: Projektverzeichnis (für den Prompt-Header).
    """
    prompt_header = f"""\
# Frame-Analyse-Auftrag

Analysiere die Bilder in `{project_dir}/frames_unique/`. Pro Bild liefere als JSON-Eintrag:

- `filename`
- `timestamp` (aus manifest.json)
- `texts` (alle wörtlich erkennbaren Texte als Liste)
- `symbols` (Symbole, Logos, Icons)
- `numbers` (alle sichtbaren Zahlen)
- `people` (erkennbare Personen)
- `places` (erkennbare Orte)
- `notable` (Auffälligkeiten)

Schreibe Ergebnis als JSON-Array nach `{project_dir}/analysis.json` UND als Markdown-Tabelle \
nach `{project_dir}/analysis.md`. Append-Modus zwischen Batches, nicht überschreiben.

"""

    timestamps = _load_manifest(manifest_file)
    batches = _build_batches(frames, batch_size)

    lines: list[str] = [prompt_header]
    for batch_num, batch in enumerate(batches, start=1):
        lines.append(f"## Batch {batch_num}\n")
        for path in batch:
            ts = timestamps.get(path.name, "")
            comment = f"  <!-- {ts} -->" if ts else ""
            lines.append(f"- {path}{comment}")
        lines.append("")

    analysis_prompt_file.write_text("\n".join(lines), encoding="utf-8")
    log.info(
        "Prompt geschrieben: %s (%d Frames, %d Batches)",
        analysis_prompt_file,
        len(frames),
        len(batches),
    )
    print(f"Prompt gespeichert: {analysis_prompt_file}")


def main() -> None:
    """Einstiegspunkt für den Prompt-Generator."""
    global log

    parser = argparse.ArgumentParser(
        description="Generiert Frame-Listen und Analyse-Prompts für Claude Code."
    )
    parser.add_argument("video_id", help="YouTube Video-ID (z.B. H0zKcbL89dU)")
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

    project_dir = OUTPUT_DIR / args.video_id
    frames_unique_dir = project_dir / "frames_unique"
    manifest_file = project_dir / "manifest.json"
    analysis_prompt_file = project_dir / "ANALYSIS_PROMPT.md"

    log = setup_logging(log_dir=project_dir)

    frames = _get_frame_paths(frames_unique_dir)
    log.info("Gefunden: %d unique Frames in %s", len(frames), project_dir)

    if args.mode == "list":
        mode_list(frames, args.batch_size)
    else:
        mode_prompt(frames, args.batch_size, manifest_file, analysis_prompt_file, project_dir)


log: logging.Logger = logging.getLogger("frame_sift")

if __name__ == "__main__":
    main()
