# frame-sift

Extrahiert alle einzigartigen Frames (Slideshow-artige Standbilder) aus einem YouTube-Video und bereitet sie für die Vision-Analyse mit Claude Code auf. Die eigentliche Analyse läuft vollständig über die Max-Plan-Quota von Claude Code — keine Anthropic-API-Calls nötig.

## Was es macht

1. Lädt ein YouTube-Video mit `yt-dlp` herunter
2. Extrahiert Frames mit `ffmpeg` im konfigurierbaren Intervall
3. Dedupliziert visuell identische/ähnliche Frames per Perceptual Hashing (pHash)
4. Erzeugt einen strukturierten Analyse-Prompt für Claude Code

## Voraussetzungen

- Python 3.11+
- `ffmpeg` im PATH
- `yt-dlp` (wird via pip installiert)
- Claude Code mit Max-Plan (für die Vision-Analyse)

## Installation

```bash
# Repository klonen
git clone https://github.com/thomas0369/frame-sift.git
cd frame-sift

# Virtuelle Umgebung einrichten
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

**ffmpeg installieren:**

| OS | Befehl |
|----|--------|
| Ubuntu/Debian | `sudo apt install ffmpeg` |
| macOS (Homebrew) | `brew install ffmpeg` |
| Windows | [ffmpeg.org/download.html](https://ffmpeg.org/download.html) |

## Workflow

### 1. Frames extrahieren

```bash
python -m src.extract "https://www.youtube.com/watch?v=BEISPIEL"
```

Erzeugt:
- `output/video.mp4` — heruntergeladenes Video
- `output/frames_raw/` — alle extrahierten Frames
- `output/frames_unique/` — deduplizierte Frames
- `output/manifest.json` — Metadaten mit Zeitstempeln

### 2. Analyse-Prompt generieren

```bash
python -m src.analyze --mode prompt
```

Erzeugt `output/ANALYSIS_PROMPT.md` mit gruppierten Frame-Listen.

### 3. Analyse in Claude Code

```bash
claude
```

Inhalt von `output/ANALYSIS_PROMPT.md` in Claude Code einfügen. Claude analysiert alle Frames und schreibt die Ergebnisse nach `output/analysis.json` und `output/analysis.md`.

**Tipp:** Bei vielen Frames `/compact` zwischen den Batches verwenden, um den Kontext zu schonen.

## Konfiguration

### `src.extract` — alle Flags

| Flag | Standard | Beschreibung |
|------|----------|--------------|
| `--fps` | `2` | Frames pro Sekunde |
| `--threshold` | `5` | pHash Hamming-Distanz (höher = aggressivere Dedup) |
| `--hash-size` | `16` | pHash-Gittergröße (16 = 16×16 Pixel) |
| `--max-height` | `1080` | Maximale Video-Auflösung |
| `--force` | — | Existierende Daten überschreiben |

### `src.analyze` — alle Flags

| Flag | Standard | Beschreibung |
|------|----------|--------------|
| `--mode` | `list` | `list`: Pfade auf stdout, `prompt`: .md erzeugen |
| `--batch-size` | `20` | Frames pro Analyse-Batch |

### Empfehlungen nach Video-Typ

| Video-Typ | Empfehlung |
|-----------|-----------|
| Slideshow / Präsentation | `--fps 1 --threshold 5` |
| Screencast / Tutorial | `--fps 2 --threshold 8` |
| Action / Bewegtbild | `--fps 4 --threshold 10` |

## Output-Struktur

```
output/
├── video.mp4              # Heruntergeladenes Video
├── frames_raw/            # Alle extrahierten Frames (frame_00001.jpg, ...)
├── frames_unique/         # Deduplizierte Frames
├── manifest.json          # Metadaten: filename, original_frame_num, timestamp_video
├── ANALYSIS_PROMPT.md     # Generierter Prompt für Claude Code
├── analysis.json          # Von Claude erzeugte Analyse (JSON-Array)
├── analysis.md            # Von Claude erzeugte Analyse (Markdown-Tabelle)
└── run.log                # Detailliertes Lauf-Protokoll
```

## Quota-Hinweise (Max-Plan)

- Das Max-Plan-Fenster beträgt **5 Stunden** pro Reset
- Empfehlung: Batches von 15–20 Frames pro Analyse-Durchgang
- `/compact` zwischen Batches in Claude Code ausführen, um den Kontext zu komprimieren
- Bei sehr langen Videos (500+ unique Frames) die Analyse auf mehrere Sessions aufteilen

## Tests ausführen

```bash
pytest
```

## Lizenz

MIT License — siehe [LICENSE](LICENSE)
