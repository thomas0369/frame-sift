from __future__ import annotations

"""Automatische Frame-Analyse via Claude Vision → analysis.json + report.md.

Primär: claude -p (Claude Code CLI, kein API-Key nötig, Max-Abo wird genutzt)
Fallback: Anthropic Python SDK (benötigt ANTHROPIC_API_KEY)
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

from src.align import align
from src.utils import OUTPUT_DIR, setup_logging

_SCHEMA = '{"texts":[],"symbols":[],"numbers":[],"people":[],"places":[],"notable":""}'

_PROMPT_TMPL = """\
Verwende ausschließlich deine Claude Vision-Fähigkeit. Kein Web-Search, keine Bash-Befehle, keine anderen Tools erlaubt — nur das Betrachten des Bildes.

Analysiere das Bild {image_path}.

Frame-Zeitstempel: {timestamp}{transcript_line}

Antworte NUR mit JSON, kein Markdown, exakt dieses Schema:
{schema}

Felder:
- texts: alle wörtlich lesbaren Texte (Liste)
- symbols: Logos, Icons, Symbole (Liste)
- numbers: alle sichtbaren Zahlen als Strings (Liste)
- people: erkennbare Personen oder Figuren (Liste)
- places: erkennbare Orte, Gebäude, Landschaften (Liste)
- notable: ein Satz über das Auffälligste im Bild (String)\
"""


def _clean_json(raw: str) -> dict:
    """Extrahiert JSON aus einer Antwort, entfernt Markdown-Fences falls nötig."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        log.warning("JSON-Parse-Fehler — leeres Ergebnis")
        return {"texts": [], "symbols": [], "numbers": [], "people": [], "places": [], "notable": ""}


def _analyze_via_claude_cli(prompt: str) -> dict:
    """Analysiert via `claude -p` — nutzt Claude Code CLI und Max-Abo.

    --allowedTools Read beschränkt Claude auf reine Vision-Analyse (kein Web, kein Bash).
    """
    result = subprocess.run(
        ["claude", "-p", "--output-format", "json", "--allowedTools", "Read", prompt],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude -p fehlgeschlagen: {result.stderr[:200]}")

    outer = json.loads(result.stdout)
    if outer.get("is_error"):
        raise RuntimeError(f"claude -p Fehler: {outer.get('result', '')[:200]}")

    return _clean_json(outer["result"])


def _analyze_via_api(client, image_path: Path, prompt_text: str) -> dict:
    """Analysiert via Anthropic Python SDK — benötigt ANTHROPIC_API_KEY."""
    import base64

    data = base64.standard_b64encode(image_path.read_bytes()).decode()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
    )
    return _clean_json(response.content[0].text)


def _build_prompt(image_path: Path, timestamp: str, transcript_text: str) -> str:
    transcript_line = f'\nGesprochener Text: "{transcript_text}"' if transcript_text else ""
    return _PROMPT_TMPL.format(
        image_path=str(image_path.resolve()),
        timestamp=timestamp,
        transcript_line=transcript_line,
        schema=_SCHEMA,
    )


def _write_report_md(project_dir: Path, entries: list[dict]) -> Path:
    lines = [
        "# Frame-Analyse-Bericht\n",
        "## Übersicht\n",
        "| Frame | Zeitstempel | Texte | Notable |",
        "|-------|-------------|-------|---------|",
    ]
    for e in entries:
        an = e.get("analysis", {})
        texts = ", ".join(an.get("texts", [])) or "—"
        notable = an.get("notable", "—")
        lines.append(f"| {e['filename']} | {e['timestamp']} | {texts} | {notable} |")

    lines.append("\n## Details\n")
    for e in entries:
        an = e.get("analysis", {})
        lines.append(f"### {e['filename']} · {e['timestamp']}")
        if e.get("transcript_text"):
            lines.append(f"> {e['transcript_text']}\n")
        for field, label in [
            ("texts", "Texte"), ("symbols", "Symbole"), ("numbers", "Zahlen"),
            ("people", "Personen"), ("places", "Orte"),
        ]:
            items = an.get(field, [])
            if items:
                lines.append(f"- **{label}:** {', '.join(str(i) for i in items)}")
        if an.get("notable"):
            lines.append(f"- **Notable:** {an['notable']}")
        lines.append("")

    out = project_dir / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def run_report(
    project_dir: Path,
    model: str = "claude-sonnet-4-6",
    force: bool = False,
    delay: float = 0.3,
) -> Path:
    """Analysiert alle Frames und schreibt analysis.json + report.md.

    Bevorzugt `claude -p` (Claude Code CLI). Fällt auf Anthropic SDK zurück
    wenn die CLI nicht verfügbar ist.
    """
    analysis_file = project_dir / "analysis.json"
    report_file = project_dir / "report.md"

    if analysis_file.exists() and report_file.exists() and not force:
        log.info("Analyse bereits vorhanden: %s (--force zum Aktualisieren)", analysis_file)
        return report_file

    manifest = json.loads((project_dir / "manifest.json").read_text(encoding="utf-8"))
    transcript_path = project_dir / "transcript.json"
    transcript = (
        json.loads(transcript_path.read_text(encoding="utf-8"))
        if transcript_path.exists()
        else {"segments": []}
    )

    timeline = align(manifest, transcript)
    frames_dir = project_dir / "frames_unique"

    # Methode wählen
    use_cli = shutil.which("claude") is not None
    api_client = None

    if use_cli:
        log.info("Analyse-Methode: claude -p (Claude Code CLI · Max-Abo)")
    else:
        log.info("claude CLI nicht gefunden — versuche Anthropic SDK")
        try:
            import anthropic, os
        except ImportError:
            log.error("anthropic SDK fehlt — pip install anthropic")
            sys.exit(1)

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            key_file = Path.home() / ".anthropic" / "api_key"
            if key_file.exists():
                api_key = key_file.read_text().strip()
        if not api_key:
            log.error(
                "Weder 'claude' CLI gefunden noch ANTHROPIC_API_KEY gesetzt.\n"
                "  Option 1 (empfohlen): Claude Code installieren — https://claude.ai/code\n"
                "  Option 2: export ANTHROPIC_API_KEY=sk-ant-..."
            )
            sys.exit(1)

        api_client = anthropic.Anthropic(api_key=api_key)

    log.info("Starte Vision-Analyse: %d Frames ...", len(timeline))
    entries = []

    for i, entry in enumerate(timeline, start=1):
        frame_path = frames_dir / entry["filename"]
        if not frame_path.exists():
            log.warning("Frame fehlt: %s — überspringe", frame_path)
            continue

        log.info("[%d/%d] %s (%s)", i, len(timeline), entry["filename"], entry["timestamp"])
        prompt = _build_prompt(frame_path, entry["timestamp"], entry["transcript_text"])

        try:
            if use_cli:
                analysis = _analyze_via_claude_cli(prompt)
            else:
                analysis = _analyze_via_api(api_client, frame_path, prompt)
        except Exception as exc:
            log.warning("Analyse fehlgeschlagen für %s: %s — überspringe", entry["filename"], exc)
            analysis = {"texts": [], "symbols": [], "numbers": [], "people": [], "places": [], "notable": ""}

        entries.append({**entry, "analysis": analysis})

        if i < len(timeline):
            time.sleep(delay)

    analysis_file.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("analysis.json: %d Einträge", len(entries))

    report = _write_report_md(project_dir, entries)
    log.info("report.md: %s", report)
    return report


def main() -> None:
    global log

    parser = argparse.ArgumentParser(
        description=(
            "Analysiert Frames via Claude Vision.\n"
            "Primär: claude -p (Claude Code CLI, Max-Abo)\n"
            "Fallback: Anthropic SDK (ANTHROPIC_API_KEY)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video_id", help="YouTube Video-ID")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Claude-Modell für SDK-Fallback (Standard: claude-sonnet-4-6)",
    )
    parser.add_argument("--delay", type=float, default=0.3, help="Pause zwischen Frames in Sekunden")
    parser.add_argument("--force", action="store_true", help="Analyse neu erstellen")
    args = parser.parse_args()

    project_dir = OUTPUT_DIR / args.video_id
    if not project_dir.exists():
        print(f"Fehler: {project_dir} nicht gefunden.", file=sys.stderr)
        sys.exit(1)

    log = setup_logging(log_dir=project_dir)
    run_report(project_dir, model=args.model, force=args.force, delay=args.delay)


log: logging.Logger = logging.getLogger("frame_sift")

if __name__ == "__main__":
    main()
