from __future__ import annotations

"""YouTube-Kanal-Scanner: Listet und filtert Videos eines Kanals."""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from src.utils import OUTPUT_DIR, setup_logging


# ---------------------------------------------------------------------------
# Video-Liste via yt-dlp
# ---------------------------------------------------------------------------

def _list_videos(channel_url: str, output_dir: Path) -> list[dict]:
    """Listet alle Videos eines Kanals via yt-dlp --flat-playlist."""
    log.info("Scanne Kanal: %s", channel_url)

    result = subprocess.run(
        [
            "yt-dlp",
            "--flat-playlist",
            "--print", "%(id)s %(title)s %(upload_date)s",
            channel_url,
            "--output", str(output_dir / "playlist.json"),
        ],
        capture_output=True,
        text=True,
    )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        log.error("Kanal-Liste konnte nicht geparst werden")
        return []


def _add_metadata(video_list: list[dict]) -> list[dict]:
    """Fügt Metadaten (Dauer, Views) zur Video-Liste hinzu."""
    log.info("Extrahiere Metadaten für %d Videos...", len(video_list))

    enriched = []
    for video in video_list:
        enriched_video = video.copy()
        enriched_video["duration_seconds"] = _extract_duration(video["id"])
        enriched_video["view_count"] = _extract_viewcount(video["id"])
        enriched.append(enriched_video)

    log.debug("Metadaten extrahiert: %d Videos", len(enriched))
    return enriched


def _extract_duration(video_id: str) -> int:
    """Extrahiert die Dauer eines Videos in Sekunden."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--print", "%(duration)s", "--get-id", video_id],
            capture_output=True,
            text=True,
            timeout=30,
        )
        parts = result.stdout.strip().split(":")
        if len(parts) >= 2:
            return int(parts[0]) * 60 + int(parts[1]) + float(parts[2]) / 60
        return 0
    except Exception:
        return 0


def _extract_viewcount(video_id: str) -> int:
    """Extrahiert die View-Anzahl eines Videos."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--print", "%(view_count)s", "--get-id", video_id],
            capture_output=True,
            text=True,
            timeout=30,
        )
        count_str = result.stdout.strip()
        count_str = count_str.replace(",", "").replace(".", "").strip()
        return int(count_str) if count_str else 0
    except Exception:
        log.warning("View-Count konnte nicht extrahiert werden für %s", video_id)
        return 0


# ---------------------------------------------------------------------------
# Filterung
# ---------------------------------------------------------------------------

def filter_videos(
    video_list: list[dict],
    min_duration: int | None = None,
    min_views: int | None = None,
    keywords: list[str] | None = None,
) -> list[dict]:
    """Filtert die Video-Liste nach Kriterien."""
    filtered = video_list

    if min_duration:
        original = len(filtered)
        filtered = [v for v in filtered if v.get("duration_seconds", 0) >= min_duration]
        removed = original - len(filtered)
        if removed > 0:
            log.info(
                "Dauer-Filter: %d Videos entfernt (< %d Sekunden)",
                removed,
                min_duration,
            )

    if min_views:
        original = len(filtered)
        filtered = [v for v in filtered if v.get("view_count", 0) >= min_views]
        removed = original - len(filtered)
        if removed > 0:
            log.info(
                "Views-Filter: %d Videos entfernt (< %d Views)",
                removed,
                min_views,
            )

    if keywords:
        original = len(filtered)
        keyword_list = [kw.lower() for kw in keywords]
        filtered = [
            v for v in filtered
            if any(kw in v.get("title", "").lower() for kw in keyword_list)
        ]
        removed = original - len(filtered)
        if removed > 0:
            log.info(
                "Schlagwort-Filter: %d Videos entfernt (%s)",
                removed,
                ", ".join(keywords),
            )

    log.info("Filter abgeschlossen: %d Videos verbleiben", len(filtered))
    return filtered


# ---------------------------------------------------------------------------
# Batch-Erstellung
# ---------------------------------------------------------------------------

def create_batches(video_list: list[dict], batch_size: int = 10) -> list[list[dict]]:
    """Teilt die Video-Liste in Batches."""
    return [
        video_list[i : i + batch_size]
        for i in range(0, len(video_list), batch_size)
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Einstiegspunkt für den Kanal-Scanner."""
    global log

    parser = argparse.ArgumentParser(
        description="Scanne und verarbeite YouTube-Kanäle",
    )
    parser.add_argument("channel_url", help="YouTube-Kanal-URL")
    parser.add_argument(
        "--min-duration",
        type=int,
        help="Minimale Dauer in Sekunden",
    )
    parser.add_argument(
        "--min-views",
        type=int,
        help="Minimale View-Anzahl",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        help="Schlagwörter im Titel (Leerzeichen-getrennt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Videos pro Batch (Standard: 10)",
    )
    args = parser.parse_args()

    log = setup_logging(log_dir=Path("."))

    # Videos auflisten
    log.info("Liste Videos aus Kanal...")
    video_list = _list_videos(args.channel_url, output_dir=Path("."))

    if not video_list:
        log.error("Keine Videos gefunden")
        sys.exit(1)

    # Metadaten hinzufügen
    video_list = _add_metadata(video_list)

    # Filtern
    filtered_videos = filter_videos(
        video_list,
        args.min_duration,
        args.min_views,
        args.keywords,
    )

    if not filtered_videos:
        log.warning("Keine Videos verbleiben nach Filterung")
        sys.exit(0)

    # Batches erstellen
    batches = create_batches(filtered_videos, args.batch_size)

    log.info(
        "Kanal-Scan abgeschlossen: %d Videos in %d Batches",
        len(filtered_videos),
        len(batches),
    )

    # Ausgabe
    for i, batch in enumerate(batches, start=1):
        print(f"Batch {i+1}: {len(batch)} Videos")
        for video in batch:
            print(f"  {video['id']} | {video['title']}")


log: logging.Logger = logging.getLogger("frame_sift")


if __name__ == "__main__":
    main()
