from __future__ import annotations

"""Transkription von YouTube-Videos: yt-dlp Untertitel → Whisper-Fallback."""

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

from src.utils import OUTPUT_DIR, setup_logging


# ---------------------------------------------------------------------------
# Zeitstempel-Hilfsfunktionen
# ---------------------------------------------------------------------------

def _ts_to_seconds(ts: str) -> float:
    """Wandelt HH:MM:SS.mmm oder HH:MM:SS,mmm in Sekunden um."""
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


def _seconds_to_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ---------------------------------------------------------------------------
# Phase 1: yt-dlp Untertitel
# ---------------------------------------------------------------------------

def _download_subtitles(url: str, project_dir: Path, lang: str) -> Path | None:
    """Lädt Untertitel via yt-dlp herunter (manuelle Subs bevorzugt, dann Auto).

    Returns:
        Pfad zur heruntergeladenen VTT-Datei oder None.
    """
    subtitle_base = project_dir / "subtitle"

    # Erst manuelle Untertitel versuchen
    for extra_flag, label in [
        ([], "manuelle"),
        (["--write-auto-subs"], "auto-generierte"),
    ]:
        cmd = [
            "yt-dlp",
            "--write-subs",
            *extra_flag,
            "--sub-langs", lang,
            "--sub-format", "vtt",
            "--skip-download",
            "--output", str(subtitle_base),
            "--no-playlist",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # yt-dlp schreibt Datei als subtitle.<lang>.vtt
        candidates = sorted(project_dir.glob("subtitle.*.vtt"))
        if candidates:
            log.info("Untertitel heruntergeladen (%s): %s", label, candidates[0].name)
            return candidates[0]

        log.debug("Keine %s Untertitel gefunden für Sprache '%s'", label, lang)

    log.info("Keine Untertitel auf YouTube verfügbar — Whisper-Fallback wird verwendet")
    return None


def _parse_vtt(vtt_path: Path) -> list[dict]:
    """Parst eine WebVTT-Datei in eine Liste von Segmenten.

    Entfernt VTT-interne Tags (<c>, Zeitmarken) und dedupliziert Wiederholungen
    aus YouTube Auto-Captions.
    """
    text = vtt_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n{2,}", text)

    ts_re = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{3})"
    )
    tag_re = re.compile(r"<[^>]+>")

    segments: list[dict] = []
    last_text = ""

    for block in blocks:
        lines = block.strip().splitlines()
        ts_line = next((l for l in lines if "-->" in l), None)
        if not ts_line:
            continue

        m = ts_re.search(ts_line)
        if not m:
            continue

        start = _ts_to_seconds(m.group(1))
        end = _ts_to_seconds(m.group(2))

        # Textzeilen nach dem Timestamp sammeln
        ts_idx = next(i for i, l in enumerate(lines) if "-->" in l)
        raw_text = " ".join(lines[ts_idx + 1 :])
        clean = tag_re.sub("", raw_text).strip()
        # HTML-Entities dekodieren
        clean = clean.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        clean = clean.replace("&nbsp;", " ").replace("&#39;", "'").replace("&quot;", '"')
        clean = re.sub(r"\s+", " ", clean).strip()

        if not clean or clean == last_text:
            continue  # Duplikat aus Auto-Captions überspringen

        # Vorheriges Segment abschliessen wenn überlappend
        if segments and segments[-1]["end"] > start:
            segments[-1]["end"] = start

        segments.append({"start": start, "end": end, "text": clean})
        last_text = clean

    return segments


# ---------------------------------------------------------------------------
# Phase 2: Whisper-Fallback
# ---------------------------------------------------------------------------

def _extract_audio(project_dir: Path, force: bool) -> Path:
    """Extrahiert Mono-Audio mit 16 kHz aus video.mp4 für Whisper."""
    audio_path = project_dir / "audio.wav"
    if audio_path.exists() and not force:
        log.info("Audio bereits vorhanden, überspringe Extraktion: %s", audio_path)
        return audio_path

    log.info("Extrahiere Audio für Whisper ...")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(project_dir / "video.mp4"),
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(audio_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log.info("Audio extrahiert: %s", audio_path)
    return audio_path


def _transcribe_whisper(
    audio_path: Path,
    lang: str | None,
    model_size: str,
    vad_filter: bool = True,
) -> list[dict]:
    """Transkribiert Audio mit faster-whisper (CTranslate2-Backend, CPU-freundlich).

    Args:
        audio_path: Pfad zur WAV-Datei.
        lang: Sprachkürzel (z.B. 'de', 'en') oder None für Auto-Erkennung.
        model_size: Whisper-Modell (tiny/base/small/medium/large).

    Returns:
        Liste von Segmenten mit start, end, text.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.error(
            "faster-whisper ist nicht installiert. Bitte ausführen:\n"
            "  pip install faster-whisper"
        )
        sys.exit(1)

    log.info("Lade Whisper-Modell '%s' (CPU) ...", model_size)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    transcribe_kwargs: dict = {"beam_size": 5, "vad_filter": vad_filter}
    if lang and lang != "auto":
        transcribe_kwargs["language"] = lang

    log.info(
        "Transkribiere mit Whisper%s%s ...",
        f" (Sprache: {lang})" if lang and lang != "auto" else " (Auto-Detect)",
        "" if vad_filter else " [VAD deaktiviert]",
    )
    segments_iter, info = model.transcribe(str(audio_path), **transcribe_kwargs)

    log.info("Erkannte Sprache: %s (%.0f%% Konfidenz)", info.language, info.language_probability * 100)

    segments = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append({"start": seg.start, "end": seg.end, "text": text})

    if not segments and vad_filter:
        log.warning(
            "Keine Sprachsegmente erkannt — das Video enthält möglicherweise "
            "keinen gesprochenen Inhalt. Mit --no-vad erneut versuchen falls doch Sprache erwartet."
        )

    return segments


# ---------------------------------------------------------------------------
# Transcript schreiben
# ---------------------------------------------------------------------------

def _write_transcript(
    project_dir: Path,
    segments: list[dict],
    source: str,
    language: str,
    model: str | None = None,
) -> Path:
    """Schreibt transcript.json mit Meta-Block und Segment-Liste.

    Jedes Segment bekommt zusätzlich ein 'timestamp'-Feld (HH:MM:SS.mmm)
    für einfache manuelle Lesbarkeit.
    """
    enriched = [
        {**seg, "timestamp": _seconds_to_ts(seg["start"])}
        for seg in segments
    ]

    meta: dict = {
        "source": source,
        "language": language,
        "segment_count": len(segments),
    }
    if model:
        meta["whisper_model"] = model

    duration = segments[-1]["end"] if segments else 0.0
    meta["duration_seconds"] = round(duration, 3)

    transcript = {"meta": meta, "segments": enriched}

    out = project_dir / "transcript.json"
    out.write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(
        "Transcript geschrieben: %s (%d Segmente, Quelle: %s)",
        out,
        len(segments),
        source,
    )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    global log

    parser = argparse.ArgumentParser(
        description=(
            "Erstellt ein Transkript für ein YouTube-Video. "
            "Versucht zuerst yt-dlp Untertitel, fällt auf Whisper zurück."
        )
    )
    parser.add_argument("video_id", help="YouTube Video-ID (z.B. H0zKcbL89dU)")
    parser.add_argument(
        "--lang",
        default="de,en",
        help="Untertitel-Sprachen für yt-dlp (Komma-getrennt) (Standard: de,en). "
             "Für Whisper: erstes Kürzel oder 'auto'",
    )
    parser.add_argument(
        "--model",
        default="large",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper-Modell (Standard: large)",
    )
    parser.add_argument(
        "--whisper-only",
        action="store_true",
        help="Yt-dlp-Schritt überspringen und direkt Whisper verwenden",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="VAD-Filter deaktivieren (Fallback falls keine Segmente erkannt werden)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Existierendes Transkript und Audio überschreiben",
    )
    args = parser.parse_args()

    project_dir = OUTPUT_DIR / args.video_id
    if not project_dir.exists():
        print(
            f"Fehler: Projektverzeichnis {project_dir} nicht gefunden.\n"
            "Bitte zuerst 'python -m src.extract <URL>' ausführen.",
            file=sys.stderr,
        )
        sys.exit(1)

    log = setup_logging(log_dir=project_dir)
    log.info("Transkription für Video-ID: %s", args.video_id)

    transcript_file = project_dir / "transcript.json"
    if transcript_file.exists() and not args.force:
        log.info("Transkript bereits vorhanden: %s (--force zum Überschreiben)", transcript_file)
        sys.exit(0)

    # Primäre Sprache aus --lang (erstes Kürzel, z.B. "de" aus "de,en")
    primary_lang = args.lang.split(",")[0].strip()

    video_url = f"https://www.youtube.com/watch?v={args.video_id}"
    segments: list[dict] = []
    source = ""

    # --- Versuch 1: yt-dlp Untertitel ---
    if not args.whisper_only:
        # Alte Subtitle-Dateien entfernen damit kein veralteter Fund
        for old in project_dir.glob("subtitle.*.vtt"):
            old.unlink()

        vtt_path = _download_subtitles(video_url, project_dir, args.lang)
        if vtt_path:
            segments = _parse_vtt(vtt_path)
            if segments:
                lang_from_file = vtt_path.stem.split(".")[-1]
                source = "youtube_subtitles"
                _write_transcript(project_dir, segments, source, lang_from_file)
                log.info(
                    "Fertig — %d Segmente aus YouTube-Untertiteln (%s)",
                    len(segments),
                    lang_from_file,
                )
                return
            log.warning("VTT geparst, aber keine Segmente — falle auf Whisper zurück")

    # --- Versuch 2: Whisper ---
    audio_path = _extract_audio(project_dir, args.force)
    whisper_lang = primary_lang if primary_lang != "auto" else None
    segments = _transcribe_whisper(audio_path, whisper_lang, args.model, vad_filter=not args.no_vad)
    source = "whisper"
    _write_transcript(project_dir, segments, source, primary_lang, model=args.model)
    log.info("Fertig — %d Segmente via Whisper (%s)", len(segments), args.model)


log: logging.Logger = logging.getLogger("frame_sift")

if __name__ == "__main__":
    main()
