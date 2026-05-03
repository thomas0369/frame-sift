from __future__ import annotations

"""Frame-Transcript-Ausrichtung: verbindet manifest.json mit transcript.json zu timeline.json."""

import json
import logging
import sys
from pathlib import Path

from src.utils import OUTPUT_DIR, setup_logging


def _ts_to_seconds(ts: str) -> float:
    """HH:MM:SS.mmm → Sekunden."""
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def align(manifest: dict, transcript: dict) -> list[dict]:
    """Verbindet jeden Frame mit zeitlich überlappenden Transkript-Segmenten.

    Sucht für jeden Frame die Segmente, in deren Zeitfenster der Frame-Zeitstempel fällt.
    Falls kein exakter Treffer: nächstes Segment innerhalb von 5 Sekunden.

    Returns:
        Timeline — eine Einheit pro Frame, mit transcript-Kontext.
    """
    segments = transcript.get("segments", [])

    timeline: list[dict] = []
    for frame in manifest["frames"]:
        frame_sec = _ts_to_seconds(frame["timestamp_video"])

        # Exakt überlappende Segmente (start ≤ frame_ts ≤ end)
        overlapping = [s for s in segments if s["start"] <= frame_sec <= s["end"]]

        # Fallback: nächstes Segment innerhalb 5 Sekunden
        if not overlapping and segments:
            close = sorted(
                (s for s in segments if abs(s["start"] - frame_sec) <= 5),
                key=lambda s: abs(s["start"] - frame_sec),
            )
            overlapping = close[:1]

        transcript_text = " ".join(s["text"] for s in overlapping)

        timeline.append(
            {
                "filename": frame["filename"],
                "timestamp": frame["timestamp_video"],
                "timestamp_seconds": round(frame_sec, 3),
                "original_frame_num": frame["original_frame_num"],
                "transcript": overlapping,
                "transcript_text": transcript_text,
            }
        )

    return timeline


def main() -> None:
    global log

    import argparse

    parser = argparse.ArgumentParser(description="Richtet Frames und Transkript zeitlich aus.")
    parser.add_argument("video_id", help="YouTube Video-ID")
    parser.add_argument("--force", action="store_true", help="timeline.json neu erstellen")
    args = parser.parse_args()

    project_dir = OUTPUT_DIR / args.video_id
    if not project_dir.exists():
        print(f"Fehler: {project_dir} nicht gefunden.", file=sys.stderr)
        sys.exit(1)

    log = setup_logging(log_dir=project_dir)

    timeline_file = project_dir / "timeline.json"
    if timeline_file.exists() and not args.force:
        log.info("Timeline bereits vorhanden: %s (--force zum Aktualisieren)", timeline_file)
        sys.exit(0)

    manifest_path = project_dir / "manifest.json"
    transcript_path = project_dir / "transcript.json"

    if not manifest_path.exists():
        log.error("manifest.json nicht gefunden — zuerst 'python -m src.extract' ausführen")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    transcript = (
        json.loads(transcript_path.read_text(encoding="utf-8"))
        if transcript_path.exists()
        else {"segments": []}
    )

    log.info(
        "Richte %d Frames mit %d Transkript-Segmenten aus ...",
        len(manifest["frames"]),
        len(transcript.get("segments", [])),
    )

    timeline = align(manifest, transcript)

    frames_with_context = sum(1 for t in timeline if t["transcript_text"])
    timeline_file.write_text(
        json.dumps({"entries": timeline}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(
        "Timeline geschrieben: %s (%d Frames, %d mit Transkript-Kontext)",
        timeline_file,
        len(timeline),
        frames_with_context,
    )


log: logging.Logger = logging.getLogger("frame_sift")

if __name__ == "__main__":
    main()
