# frame-sift — Architektur

## Gesamtpipeline

```mermaid
flowchart TD
    USER(["Benutzer"]) -->|"YouTube-URL"| EXTRACT["python -m src.extract"]

    subgraph "Phase 1: Extraktion (src/extract.py)"
        EXTRACT --> VALIDATE["validate_dependencies()<br/>yt-dlp + ffmpeg prüfen"]
        VALIDATE --> DL["_download_video()<br/>yt-dlp → output/video.mp4"]
        DL --> FRAMES["_extract_frames()<br/>ffmpeg → output/frames_raw/*.jpg"]
        FRAMES --> DEDUP["_run_dedup()<br/>pHash Deduplizierung"]
        DEDUP --> MANIFEST["_write_manifest()<br/>output/manifest.json"]
    end

    MANIFEST -->|"unique Frames vorhanden"| ANALYZE["python -m src.analyze --mode prompt"]

    subgraph "Phase 2: Prompt-Generierung (src/analyze.py)"
        ANALYZE --> LOADM["_load_manifest()<br/>{filename → timestamp}"]
        LOADM --> GETF["_get_frame_paths()<br/>output/frames_unique/*.jpg"]
        GETF --> BATCH["_build_batches()<br/>N Frames pro Batch"]
        BATCH --> PROMPT["mode_prompt()<br/>output/ANALYSIS_PROMPT.md"]
    end

    PROMPT -->|"Prompt einfügen"| CLAUDE["Claude Code<br/>(Vision-Analyse)"]

    subgraph "Phase 3: Analyse (Claude Code)"
        CLAUDE --> JSON["output/analysis.json<br/>JSON-Array"]
        CLAUDE --> MD["output/analysis.md<br/>Markdown-Tabelle"]
    end

    style VALIDATE fill:#369,stroke:#333,color:#fff
    style DEDUP fill:#2d6,stroke:#333,color:#000
    style CLAUDE fill:#a36,stroke:#333,color:#fff
```

## Deduplizierungs-Algorithmus (Perceptual Hash)

```mermaid
flowchart LR
    START(["Frame-Liste<br/>sortiert"]) --> LOOP["Für jeden Frame"]
    LOOP --> OPEN["PIL Image öffnen"]
    OPEN --> HASH["imagehash.phash(img,<br/>hash_size=N)"]
    HASH --> COMPARE{"Hamming-Distanz<br/>zu allen kept_hashes<br/>≤ threshold?"}
    COMPARE -->|"Ja: Duplikat"| SKIP["Frame überspringen"]
    COMPARE -->|"Nein: Einzigartig"| KEEP["kept_indices.append(idx)<br/>kept_hashes.append(h)"]
    KEEP --> NEXT["Nächster Frame"]
    SKIP --> NEXT
    NEXT --> LOOP
    NEXT -->|"Alle verarbeitet"| COPY["Einzigartige Frames<br/>nach frames_unique/ kopieren"]
    COPY --> END(["Manifest schreiben"])

    style KEEP fill:#2d6,stroke:#333,color:#000
    style SKIP fill:#d33,stroke:#333,color:#fff
```

## Datenmodell (manifest.json)

```mermaid
erDiagram
    MANIFEST_ENTRY {
        TEXT filename PK "frame_NNNNN.jpg"
        INTEGER original_frame_num "1-basierte ffmpeg-Nummer"
        TEXT timestamp_video "HH:MM:SS.mmm"
    }

    ANALYSIS_ENTRY {
        TEXT filename FK "Referenz auf MANIFEST_ENTRY"
        TEXT timestamp "Kopiiert aus Manifest"
        JSON texts "Erkannte Texte"
        JSON symbols "Symbole, Logos"
        JSON numbers "Sichtbare Zahlen"
        JSON people "Erkennbare Personen"
        JSON places "Erkennbare Orte"
        TEXT notable "Auffälligkeiten"
    }

    MANIFEST_ENTRY ||--o| ANALYSIS_ENTRY : "analysiert als"
```

## Output-Verzeichnisstruktur

```mermaid
flowchart TD
    OUT["output/"]
    OUT --> VID["video.mp4<br/>yt-dlp Download"]
    OUT --> RAW["frames_raw/<br/>frame_00001.jpg … frame_NNNNN.jpg<br/>alle ffmpeg-Frames"]
    OUT --> UNQ["frames_unique/<br/>frame_00001.jpg … frame_MMMMM.jpg<br/>dedupliziert (M ≤ N)"]
    OUT --> MFT["manifest.json<br/>filename + orig_num + timestamp"]
    OUT --> APF["ANALYSIS_PROMPT.md<br/>Batched-Prompt für Claude Code"]
    OUT --> AJS["analysis.json<br/>Claude Code Output"]
    OUT --> AMD["analysis.md<br/>Claude Code Output"]
    OUT --> LOG["run.log<br/>Detailliertes Protokoll"]

    style RAW fill:#555,stroke:#333,color:#fff
    style UNQ fill:#2d6,stroke:#333,color:#000
    style AJS fill:#a36,stroke:#333,color:#fff
    style AMD fill:#a36,stroke:#333,color:#fff
```

## Cache-Verhalten (Idempotenz)

```mermaid
stateDiagram-v2
    [*] --> CHECK_VIDEO: extract gestartet

    CHECK_VIDEO --> DOWNLOAD: video.mp4 fehlt
    CHECK_VIDEO --> CHECK_FRAMES: video.mp4 vorhanden<br/>und --force nicht gesetzt

    DOWNLOAD --> CHECK_FRAMES: Download OK

    CHECK_FRAMES --> EXTRACT_FRAMES: frames_raw/ leer
    CHECK_FRAMES --> CHECK_UNIQUE: frames_raw/ vorhanden<br/>und --force nicht gesetzt

    EXTRACT_FRAMES --> CHECK_UNIQUE: Extraktion OK

    CHECK_UNIQUE --> DEDUP: frames_unique/ leer
    CHECK_UNIQUE --> WRITE_MANIFEST: frames_unique/ vorhanden<br/>und --force nicht gesetzt

    DEDUP --> WRITE_MANIFEST: Dedup OK

    WRITE_MANIFEST --> [*]: manifest.json geschrieben

    note right of CHECK_VIDEO
        --force überspringt alle<br/>Cache-Checks
    end note
```

## Konfigurationsparameter

```mermaid
flowchart LR
    subgraph "src.extract"
        FPS["--fps (Standard: 2.0)<br/>ffmpeg fps-Filter"]
        THR["--threshold (Standard: 5)<br/>Hamming-Distanz-Schwelle"]
        HS["--hash-size (Standard: 16)<br/>pHash NxN-Gitter"]
        MH["--max-height (Standard: 1080)<br/>yt-dlp Formatfilter"]
        FRC["--force<br/>Cache-Bypass"]
    end

    subgraph "src.analyze"
        MODE["--mode (list | prompt)<br/>Ausgabeformat"]
        BS["--batch-size (Standard: 20)<br/>Frames pro Batch"]
    end

    THR -->|"höher = aggressiver"| DEDUP_OUT["Mehr Frames<br/>werden verworfen"]
    FPS -->|"höher = mehr Frames"| RAW_OUT["Größere frames_raw/"]
    BS -->|"kleiner = mehr Batches"| QUOTA_OUT["Schont Claude-Quota<br/>pro Session"]
```
