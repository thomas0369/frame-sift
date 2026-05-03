# frame-sift — Architektur

## Gesamtpipeline

```mermaid
flowchart TD
    USER(["Benutzer"]) -->|"YouTube-URL"| PIPE["python -m src.pipeline &lt;URL&gt;\nEinziger Einstiegspunkt"]
    USER -->|"manuell"| EXTRACT["python -m src.extract &lt;URL&gt;"]

    PIPE --> EXTRACT
    PIPE --> TRANSCRIBE["python -m src.transcribe &lt;video_id&gt;"]
    PIPE --> ALIGN["python -m src.align &lt;video_id&gt;"]
    PIPE --> REPORT["python -m src.report &lt;video_id&gt;"]

    subgraph "Modul 1: Extraktion (src/extract.py)"
        EXTRACT --> VT["Video-Typ-Erkennung\n_presample_for_type() + _detect_video_type()\nSlideshow → 1 fps · Live-Action → 2 fps"]
        VT --> DL["_download_video()\nyt-dlp → output/{video_id}/video.mp4"]
        DL --> FR["_extract_frames()\nffmpeg adaptive fps → frames_raw/"]
        FR --> PH["_compute_frame_hashes()\nProcessPoolExecutor ≥80 Frames"]
        PH --> TH["_detect_threshold() Valley Detection"]
        TH --> D1["_run_dedup() Pass 1 · Sliding-Window"]
        D1 --> D2["_run_dedup_global() Pass 2 · Global"]
        D2 --> MF["manifest.json"]
    end

    subgraph "Modul 2: Transkription (src/transcribe.py)"
        TRANSCRIBE --> SUB["yt-dlp Untertitel → VTT"]
        SUB -->|"vorhanden"| PARSE["_parse_vtt() · Dedup"]
        SUB -->|"fehlt"| WH["faster-whisper large · VAD"]
        PARSE & WH --> TR["transcript.json"]
    end

    subgraph "Modul 3: Ausrichtung (src/align.py)"
        ALIGN --> ALG["align(manifest, transcript)\nFrame-Sekunde ↔ Segment-Fenster"]
        ALG --> TL["timeline.json\n{filename, timestamp, transcript_text}"]
    end

    subgraph "Modul 4: Vision-Analyse (src/report.py)"
        REPORT --> API["Claude Vision API\nclaude-sonnet-4-6\n1 Request / Frame"]
        API --> AJ["analysis.json"]
        API --> RM["report.md\nTabelle + Detailabschnitte"]
    end

    MF --> ALIGN
    TR --> ALIGN
    TL --> REPORT
    RM --> SEARCH["python -m src.search &lt;video_id&gt; &lt;Begriff&gt;"]

    style PIPE fill:#a36,stroke:#333,color:#fff
    style PH fill:#365,stroke:#333,color:#fff
    style D2 fill:#365,stroke:#333,color:#fff
    style WH fill:#563,stroke:#333,color:#fff
    style API fill:#a36,stroke:#333,color:#fff
    style SEARCH fill:#246,stroke:#333,color:#fff
```

## Extraktion: Detaillierter Ablauf

```mermaid
flowchart TD
    START(["src.extract URL"]) --> PRE["Pre-Sample\n10 Frames bei 1 fps\n_presample_for_type()"]
    PRE --> TYPE{"Video-Typ?"}
    TYPE -->|"&gt;60% low-dist"| SLIDE["slideshow\n→ 1.0 fps"]
    TYPE -->|"&lt;20% low-dist"| LIVE["live-action\n→ 2.0 fps"]
    TYPE -->|"sonst"| MIX["mixed\n→ 2.0 fps"]
    SLIDE & LIVE & MIX --> EX["ffmpeg Extraktion\nframes_raw/frame_NNNNN.jpg"]

    EX --> NH{"≥ 80 Frames?"}
    NH -->|"Ja"| PAR["ProcessPoolExecutor\n_hash_single_frame(path, hash_size)\n→ Hex-String serialisiert"]
    NH -->|"Nein"| SEQ["Sequentielles Hashing\nimagehash.phash()"]
    PAR -->|"Fehler"| SEQ
    PAR & SEQ --> HH["Hashes: list[ImageHash | None]"]

    HH --> AUTOTH{"--threshold\nangegeben?"}
    AUTOTH -->|"Nein"| VAL["_detect_threshold()\nValley Detection\nauf Raw-Frame-Distanzen"]
    AUTOTH -->|"Ja"| MANU["Manueller Threshold"]
    VAL & MANU --> MODE{"--dedup-mode?"}

    MODE -->|"sliding"| SL["_dedupe_by_hash_sliding()\njeder Frame vs. Vorgänger-Raw-Frame\n→ misst momentane Änderung"]
    MODE -->|"global"| GL["_dedupe_by_hash()\njeder Frame vs. alle kept_hashes"]
    SL --> P2{"--no-second-pass?"}
    P2 -->|"Nein"| DG["_run_dedup_global() · Pass 2\n_detect_pass2_threshold(): Gap-Analyse\n→ Globaler Dedup auf Unique-Frames"]
    P2 -->|"Ja"| CP["frames kopieren\n→ frames_unique/"]
    DG --> CP
    GL --> CP
    CP --> MW["_write_manifest()"]
    MW --> DONE(["manifest.json"])

    style PAR fill:#365,stroke:#333,color:#fff
    style SL fill:#2d6,stroke:#333,color:#000
    style DG fill:#2d6,stroke:#333,color:#000
    style VAL fill:#2d6,stroke:#333,color:#000
```

## Transkription: Fallback-Strategie

```mermaid
flowchart TD
    START(["src.transcribe video_id"]) --> SKIP{"--whisper-only?"}
    SKIP -->|"Nein"| YT["yt-dlp --write-subs\nManuelle Untertitel · VTT"]
    YT --> MAN{"Datei vorhanden?"}
    MAN -->|"Nein"| YTA["yt-dlp --write-auto-subs\nAuto-generierte Untertitel"]
    YTA --> AUTO{"Datei vorhanden?"}
    AUTO -->|"Nein"| WFALLBACK["Whisper-Fallback"]
    AUTO -->|"Ja"| PARSE["_parse_vtt()\nTag-Stripping + Dedup"]
    MAN -->|"Ja"| PARSE
    PARSE --> SEG{"Segmente > 0?"}
    SEG -->|"Nein"| WFALLBACK
    SEG -->|"Ja"| DONE

    SKIP -->|"Ja"| WFALLBACK
    WFALLBACK --> AUD["_extract_audio()\nffmpeg → audio.wav\n16 kHz · Mono · PCM"]
    AUD --> WH["faster-whisper\nWhisperModel(model, device='cpu', int8)\nvad_filter=True"]
    WH --> SEGS{"Segmente > 0?"}
    SEGS -->|"Nein"| WARN["Warnung: kein Sprachinhalt\n→ leeres transcript.json"]
    SEGS -->|"Ja"| DONE
    WARN --> DONE

    DONE(["_write_transcript()\ntranscript.json"])

    style WH fill:#563,stroke:#333,color:#fff
    style PARSE fill:#2d6,stroke:#333,color:#000
```

## Datenmodell

```mermaid
erDiagram
    MANIFEST_META {
        INTEGER threshold_used "Auto-erkannt oder manuell"
        BOOLEAN auto_threshold
        TEXT dedup_mode "sliding | global"
        INTEGER hash_size "Standard: 16"
        FLOAT fps "1.0 oder 2.0, adaptiv"
        TEXT video_type "slideshow | mixed | live-action"
        INTEGER raw_frame_count
        INTEGER unique_frame_count
        INTEGER pass1_count "nach Sliding-Pass"
        INTEGER pass2_count "nach Global-Pass (optional)"
    }

    MANIFEST_FRAME {
        TEXT filename PK "frame_NNNNN.jpg"
        INTEGER original_frame_num "ffmpeg-Nummer (1-basiert)"
        TEXT timestamp_video "HH:MM:SS.mmm"
    }

    TRANSCRIPT_META {
        TEXT source "youtube_subtitles | whisper"
        TEXT language "de | en | …"
        TEXT whisper_model "large | medium | …"
        INTEGER segment_count
        FLOAT duration_seconds
    }

    TRANSCRIPT_SEGMENT {
        FLOAT start "Sekunden"
        FLOAT end "Sekunden"
        TEXT text "Transkribierter Inhalt"
        TEXT timestamp "HH:MM:SS.mmm"
    }

    MANIFEST_META ||--|{ MANIFEST_FRAME : "enthält"
    TRANSCRIPT_META ||--|{ TRANSCRIPT_SEGMENT : "enthält"
    MANIFEST_FRAME }o--o{ TRANSCRIPT_SEGMENT : "zeitlich ausgerichtet"
```

## Output-Verzeichnisstruktur (pro Video)

```mermaid
flowchart TD
    OUT["output/"]
    OUT --> VID_DIR["{video_id}/\nz.B. H0zKcbL89dU/"]

    VID_DIR --> MP4["video.mp4\nyt-dlp Download"]
    VID_DIR --> RAW["frames_raw/\nframe_00001.jpg … frame_NNNNN.jpg\nalle ffmpeg-Frames · adaptive fps"]
    VID_DIR --> UNQ["frames_unique/\nframe_00001.jpg … frame_MMMMM.jpg\nnach Dedup (M ≤ N)"]
    VID_DIR --> MFT["manifest.json\nmeta + frames[]"]
    VID_DIR --> TRS["transcript.json\nmeta + segments[]"]
    VID_DIR --> WAV["audio.wav\n16 kHz Mono · nur bei Whisper-Fallback"]
    VID_DIR --> VTT["subtitle.&lt;lang&gt;.vtt\nDownload-Cache"]
    VID_DIR --> APF["ANALYSIS_PROMPT.md\nBatched-Prompt für Claude Code"]
    VID_DIR --> LOG["run.log\nDetailliertes Protokoll"]

    style RAW fill:#555,stroke:#333,color:#fff
    style UNQ fill:#2d6,stroke:#333,color:#000
    style TRS fill:#563,stroke:#333,color:#fff
```

## Cache-Verhalten (Idempotenz)

```mermaid
stateDiagram-v2
    [*] --> CHECK_VIDEO: extract gestartet

    CHECK_VIDEO --> PRESAMPLE: video.mp4 vorhanden
    CHECK_VIDEO --> DOWNLOAD: video.mp4 fehlt
    DOWNLOAD --> PRESAMPLE: Download OK

    PRESAMPLE --> CHECK_FRAMES: frames_raw/ vorhanden + kein --force
    PRESAMPLE --> DETECT_TYPE: Pre-Sample (10 Frames)
    DETECT_TYPE --> EXTRACT_FRAMES: Typ + FPS erkannt
    CHECK_FRAMES --> HASH: Cache-Treffer

    EXTRACT_FRAMES --> HASH: Extraktion OK

    HASH --> HASH_PARALLEL: raw_count ≥ 80
    HASH --> HASH_SEQ: raw_count < 80
    HASH_PARALLEL --> THRESHOLD
    HASH_SEQ --> THRESHOLD
    HASH_PARALLEL --> HASH_SEQ: Pool-Fehler

    THRESHOLD --> CHECK_UNIQUE

    CHECK_UNIQUE --> PASS1: frames_unique/ leer oder --force
    CHECK_UNIQUE --> DONE: frames_unique/ vorhanden + kein --force

    PASS1 --> PASS2: dedup_mode=sliding + kein --no-second-pass
    PASS1 --> COPY: dedup_mode=global oder --no-second-pass
    PASS2 --> WARN_FEW: unique_count < 5
    WARN_FEW --> COPY
    PASS2 --> COPY: Pass 2 OK

    COPY --> MANIFEST
    MANIFEST --> DONE

    DONE --> [*]: manifest.json aktuell

    note right of HASH_PARALLEL
        ProcessPoolExecutor
        _hash_single_frame() top-level
        Hex-Serialisierung (picklable)
    end note
```

## Konfigurationsparameter

```mermaid
flowchart LR
    subgraph "src.extract"
        FPS["--fps\nStandard: auto\n1.0 Slideshow · 2.0 sonst"]
        THR["--threshold\nStandard: auto\nValley Detection"]
        HS["--hash-size 16\npHash NxN-Gitter"]
        MH["--max-height 1080\nyt-dlp Formatfilter"]
        DM["--dedup-mode sliding|global\nStandard: sliding"]
        NSP["--no-second-pass\nPass 2 überspringen"]
        FRC["--force\nCache-Bypass"]
    end

    subgraph "src.transcribe"
        LANG["--lang de,en\nSprachen für yt-dlp + Whisper"]
        MOD["--model large\ntiny/base/small/medium/large"]
        WO["--whisper-only\nyt-dlp überspringen"]
        NOVAD["--no-vad\nVAD-Filter deaktivieren"]
        TFRC["--force\nTranskript neu erstellen"]
    end

    subgraph "src.analyze"
        MODE["--mode list|prompt\nAusgabeformat"]
        BS["--batch-size 20\nFrames pro Batch"]
    end

    THR -->|"None → auto"| AUTOT["Valley Detection\nerster Tal-Bucket"]
    FPS -->|"None → auto"| AUTOF["Pre-Sample\n→ Typ-Erkennung"]
    MOD -->|"höher = langsamer"| QUAL["Bessere\nTranskriptqualität"]
    BS -->|"kleiner = mehr Batches"| QUOTA["Schont Claude-Quota\npro Session"]
```
