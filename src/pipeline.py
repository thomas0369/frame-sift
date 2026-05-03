from __future__ import annotations

"""Einziger Einstiegspunkt: extract → transcribe → align → report."""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from src.extract import _extract_video_id
from src.utils import OUTPUT_DIR, setup_logging


def _run(description: str, module: str, args: list[str]) -> bool:
    """Führt ein Untermodul als Subprozess aus und gibt True bei Erfolg zurück."""
    cmd = [sys.executable, "-m", module, *args]
    log.info("── %s", description)
    result = subprocess.run(cmd, cwd=Path.cwd())
    if result.returncode != 0:
        log.error("%s fehlgeschlagen (exit %d)", description, result.returncode)
        return False
    return True


def main() -> None:
    global log

    parser = argparse.ArgumentParser(
        description=(
            "frame-sift vollständige Pipeline: "
            "Extract → Transcribe → Align → Report (Oder Kanal-Scan)"
        )
    )
    parser.add_argument("url", help="YouTube-URL oder Kanal-URL")
    parser.add_argument(
        "--lang",
        default="de,en",
        help="Untertitel-/Transkript-Sprachen (Standard: de,en)",
    )
    parser.add_argument(
        "--whisper-model",
        default="large",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper-Modell für Transkription (Standard: large)",
    )
    parser.add_argument(
        "--vision-model",
        default="claude-sonnet-4-6",
        help="Claude-Modell für Vision-Analyse (Standard: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="Transkriptions-Schritt überspringen",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Vision-Analyse-Schritt überspringen",
    )
    parser.add_argument(
        "--scan-channel",
        action="store_true",
        help="Kanal-Scannen-Modus (liest Videos, ruft Pipeline für jedes)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Alle Schritte neu ausführen (ignoriert Cache)",
    )
    args = parser.parse_args()

    force_flag = ["--force"] if args.force else []

    # Kanal-Scan-Modus
    if args.scan_channel:
        log.info("Kanal-Scan-Modus aktiviert")

        from src.channel import (
            _list_videos,
            _add_metadata,
            filter_videos,
            create_batches,
        )

        output_dir = Path(".")
        channel_url = args.url

        # Videos auflisten
        log.info("Liste Videos aus Kanal...")
        video_list = _list_videos(channel_url, output_dir=output_dir)

        if not video_list:
            log.error("Keine Videos gefunden")
            sys.exit(1)

        # Metadaten hinzufügen
        video_list = _add_metadata(video_list)

        # Filtern
        filtered_videos = filter_videos(
            video_list,
            min_duration=None,
            min_views=None,
            keywords=None,
        )

        if not filtered_videos:
            log.warning("Keine Videos verbleiben nach Filterung")
            sys.exit(0)

        # Batches erstellen
        batches = create_batches(filtered_videos, batch_size=10)

        log.info(
            "Kanal-Scan abgeschlossen: %d Videos in %d Batches",
            len(filtered_videos),
            len(batches),
        )

        for i, batch in enumerate(batches, start=1):
            print(f"Batch {i+1}: {len(batch)} Videos")
        for video in batch:
            print(f"  {video['id']} | {video['title']}")

        sys.exit(0)

    # Normaler Pipeline-Modus (einzelnes Video)
    video_id = _extract_video_id(args.url)
    project_dir = OUTPUT_DIR / video_id
    project_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(log_dir=project_dir)
    t_start = time.monotonic()

    log.info("═══════════════════════════")
    log.info("frame-sift Pipeline · Video-ID: %s", video_id)
    log.info("═════════════════════════════")

    # Schritt 1: Extract
    ok = _run(
        "Schritt 1/4: Frame-Extraktion",
        "src.extract",
        [args.url, *force_flag],
    )
    if not ok:
        sys.exit(1)

    # Schritt 2: Transcribe
    if not args.skip_transcribe:
        ok = _run(
            "Schritt 2/4: Transkription",
            "src.transcribe",
            [video_id, "--lang", args.lang, "--model", args.whisper_model, *force_flag],
        )
        if not ok:
            log.warning("Transkription fehlgeschlagen — Pipeline läuft ohne Transkript weiter")
    else:
        log.info("── Schritt 2/4: Transkription übersprungen (--skip-transcribe)")

    # Schritt 3: Align
    ok = _run(
        "Schritt 3/4: Frame-Transcript-Ausrichtung",
        "src.align",
        [video_id, *force_flag],
    )
    if not ok:
        log.warning("Ausrichtung fehlgeschlagen — Pipeline läuft ohne Timeline weiter")
    else:
        log.info("── Schritt 3/4: Ausrichtung abgeschlossen")

    # Schritt 4: Report
    if not args.skip_report:
        ok = _run(
            "Schritt 4/4: Vision-Analyse",
            "src.report",
            [video_id, "--model", args.vision_model, *force_flag],
        )
        if not ok:
            log.error("Vision-Analyse fehlgeschlagen")
            sys.exit(1)
    else:
        log.info("── Schritt 4/4: Vision-Analyse übersprungen (--skip-report)")

    elapsed = time.monotonic() - t_start
    log.info("═══════════════════════════")
    log.info("Fertig in %.1fs — Ergebnisse in %s/", elapsed, project_dir)
    log.info("  manifest.json  · transcript.json  · timeline.json")
    log.info("  analysis.json  · report.md")
    log.info("  Suche: python -m src.search %s <Begriff>", video_id)
    log.info("═════════════════════════════")


log: logging.Logger = logging.getLogger("frame_sift")

if __name__ == "__main__":
    main()
