from __future__ import annotations

"""Volltextsuche über analysis.json und transcript.json eines Projekts."""

import argparse
import json
import re
import sys
from pathlib import Path

from src.utils import OUTPUT_DIR


def _highlight(text: str, pattern: re.Pattern) -> str:
    return pattern.sub(lambda m: f"**{m.group()}**", text)


def search(project_dir: Path, query: str) -> list[dict]:
    """Durchsucht analysis.json und transcript.json nach query.

    Returns:
        Liste von Treffern mit Herkunft, Zeitstempel und Kontext.
    """
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    hits: list[dict] = []

    # --- analysis.json ---
    analysis_path = project_dir / "analysis.json"
    if analysis_path.exists():
        entries = json.loads(analysis_path.read_text(encoding="utf-8"))
        for entry in entries:
            an = entry.get("analysis", {})
            matched_fields: list[str] = []

            for field in ("texts", "symbols", "numbers", "people", "places"):
                for val in an.get(field, []):
                    if pattern.search(str(val)):
                        matched_fields.append(f"{field}: {_highlight(str(val), pattern)}")

            notable = an.get("notable", "")
            if notable and pattern.search(notable):
                matched_fields.append(f"notable: {_highlight(notable, pattern)}")

            transcript = entry.get("transcript_text", "")
            if transcript and pattern.search(transcript):
                matched_fields.append(f"transkript: {_highlight(transcript, pattern)}")

            if matched_fields:
                hits.append(
                    {
                        "source": "analysis",
                        "filename": entry["filename"],
                        "timestamp": entry["timestamp"],
                        "matches": matched_fields,
                    }
                )

    # --- transcript.json (auch wenn keine analysis vorhanden) ---
    transcript_path = project_dir / "transcript.json"
    if transcript_path.exists() and not analysis_path.exists():
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
        for seg in data.get("segments", []):
            if pattern.search(seg.get("text", "")):
                hits.append(
                    {
                        "source": "transcript",
                        "timestamp": seg.get("timestamp", ""),
                        "matches": [_highlight(seg["text"], pattern)],
                    }
                )

    return hits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sucht in der Frame-Analyse und dem Transkript eines Videos."
    )
    parser.add_argument("video_id", help="YouTube Video-ID")
    parser.add_argument("query", help="Suchbegriff")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Ausgabe als JSON",
    )
    args = parser.parse_args()

    project_dir = OUTPUT_DIR / args.video_id
    if not project_dir.exists():
        print(f"Fehler: {project_dir} nicht gefunden.", file=sys.stderr)
        sys.exit(1)

    hits = search(project_dir, args.query)

    if args.as_json:
        print(json.dumps(hits, indent=2, ensure_ascii=False))
        return

    if not hits:
        print(f'Keine Treffer für "{args.query}" in {args.video_id}.')
        return

    print(f'{len(hits)} Treffer für "{args.query}" in {args.video_id}:\n')
    for h in hits:
        label = h.get("filename", "transkript")
        ts = h.get("timestamp", "")
        print(f"  {label}  [{ts}]")
        for m in h["matches"]:
            print(f"    → {m}")
        print()
