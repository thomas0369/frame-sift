from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path


OUTPUT_DIR = Path("output")
LOG_FILE = OUTPUT_DIR / "run.log"


def setup_logging() -> logging.Logger:
    """Richtet strukturiertes Logging nach stdout und output/run.log ein.

    Returns:
        Konfigurierter Logger für das Paket.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    fmt = "%(asctime)s %(levelname)-8s %(message)s"
    datefmt = "%H:%M:%S"

    logger = logging.getLogger("frame_sift")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

        logger.addHandler(console)
        logger.addHandler(file_handler)

    return logger


def frame_to_timestamp(frame_num: int, fps: float) -> str:
    """Wandelt eine Frame-Nummer in einen Video-Zeitstempel um.

    Args:
        frame_num: 1-basierte Frame-Nummer (wie von ffmpeg erzeugt).
        fps: Frames pro Sekunde, mit der die Frames extrahiert wurden.

    Returns:
        Zeitstempel im Format HH:MM:SS.mmm.
    """
    total_seconds = (frame_num - 1) / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    return f"{hours:02d}:{minutes:02d}:{whole:02d}.{millis:03d}"


def validate_dependencies() -> None:
    """Prüft, ob yt-dlp und ffmpeg im PATH vorhanden sind.

    Raises:
        SystemExit: Wenn eine oder beide Abhängigkeiten fehlen.
    """
    missing: list[str] = []

    for tool in ("yt-dlp", "ffmpeg"):
        if shutil.which(tool) is None:
            missing.append(tool)

    if missing:
        tools_str = " und ".join(missing)
        print(
            f"Fehler: {tools_str} nicht im PATH gefunden.\n"
            "Installation:\n"
            "  ffmpeg:  https://ffmpeg.org/download.html\n"
            "  yt-dlp:  pip install yt-dlp",
            file=sys.stderr,
        )
        sys.exit(1)
