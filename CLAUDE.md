# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projektübersicht

frame-sift extrahiert einzigartige Frames aus YouTube-Videos, erstellt Transkripte und bereitet alles für die Vision-Analyse mit Claude auf. Die Pipeline: Download → Frame-Extraktion → pHash-Dedup → Transkription → Ausrichtung → Vision-Analyse.

## Commands

```bash
# Tests
.venv/bin/python -m pytest                          # alle Tests
.venv/bin/python -m pytest tests/test_dedup.py      # einzelne Datei
.venv/bin/python -m pytest tests/test_hashing.py::TestHashSingleFrame::test_valid_image_returns_hex  # einzelner Test

# Pipeline
.venv/bin/python -m src.pipeline <URL>              # vollständige Pipeline
.venv/bin/python -m src.extract <URL>               # nur Frame-Extraktion
.venv/bin/python -m src.transcribe <video_id>       # nur Transkription
.venv/bin/python -m src.align <video_id>            # nur Ausrichtung
.venv/bin/python -m src.report <video_id>           # nur Vision-Analyse
.venv/bin/python -m src.search <video_id> <Begriff> # Volltextsuche
```

Externe Abhängigkeiten: `ffmpeg` und `yt-dlp` müssen im PATH sein.

## Architektur

Jedes Modul in `src/` ist eigenständig über `python -m src.<modul>` aufrufbar. `src/pipeline.py` ist der einzige Orchestrator, der die Module als Subprozesse aufruft.

**Datenfluss:** Jedes Video landet in `output/<video_id>/`. Zwischenprodukte (manifest.json, transcript.json, timeline.json) sind JSON-Dateien, die von nachfolgenden Modulen gelesen werden.

**Dedup-Algorithmus (src/extract.py):** Zweistufig. Pass 1 nutzt Sliding-Window (jeder Frame vs. Vorgänger — erfasst Ken-Burns-Effekte). Pass 2 läuft global auf den verbleibenden Unique-Frames und nutzt Gap-Analyse auf sortierten Distanzen. Der Threshold wird automatisch per Valley Detection erkannt.

**Video-Typ-Erkennung:** Pre-Sample (10 Frames bei 1 fps) → pHash → Anteil low-distance Paare. Slideshow (>60%) → 1 fps, Live-Action (<20%) → 2 fps.

**Transkription (src/transcribe.py):** Fallback-Kette: YouTube-Manual-Subs → Auto-Subs (VTT) → faster-whisper (CPU, int8). VTT-Parser dedupliziert Auto-Caption-Wiederholungen und entfernt VTT-Tags/HTML-Entities.

**Ausrichtung (src/align.py):** Verbindet Frames (manifest.json) mit Transkript-Segmenten (transcript.json) über Zeitstempel-Overlap. Fallback: nächstes Segment innerhalb ±5 Sekunden.

**Vision-Analyse (src/report.py):** Primär `claude -p` (CLI, Max-Abo), Fallback: Anthropic Python SDK (`ANTHROPIC_API_KEY`). Pro Frame ein API-Call mit Structured-Output-Schema.

## Wichtige Patterns

- **Cache/Idempotenz:** Alle Module überspringen bestehende Ausgaben, es sei denn `--force` ist gesetzt. Das gilt für Downloads, Frames, Transkripte, Timeline, Analyse.
- **Logging:** `src/utils.py:setup_logging()` konfiguriert dual (stdout INFO + `run.log` DEBUG). Module-Level `log` wird in `main()` per `global log` zugewiesen.
- **Module-Logger patchen:** Tests patchen den Logger mit `monkeypatch.setattr(mod, "log", logging.getLogger("test"))` weil Module-Level `log` initial `logging.getLogger("frame_sift")` ist und erst in `main()` neu zugewiesen wird.
- **Hashing parallelisieren:** `_hash_single_frame()` ist top-level (picklable für ProcessPoolExecutor). Ab 80 Frames wird parallelisiert, Fallback auf sequentiell.
- **Manifest-Format:** `{"meta": {...}, "frames": [{filename, original_frame_num, timestamp_video}]}`. Timestamps basieren immer auf `orig_raw_paths`, nie auf umnummerierten Unique-Frame-Namen.

## Sprachkonvention

Code und Kommentare sind auf Deutsch. Commit-Messages und Dokumentation ebenfalls.
