from __future__ import annotations

"""Tests für src/transcribe.py — VTT-Parser, Timestamp-Konvertierung, Transcript-Format."""

import json
from pathlib import Path

import pytest

from src.transcribe import _parse_vtt, _seconds_to_ts, _ts_to_seconds, _write_transcript


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vtt(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "sub.vtt"
    p.write_text("WEBVTT\n\n" + content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _ts_to_seconds
# ---------------------------------------------------------------------------

class TestTsToSeconds:
    def test_basic(self) -> None:
        assert _ts_to_seconds("00:01:30.500") == pytest.approx(90.5)

    def test_comma_separator(self) -> None:
        assert _ts_to_seconds("00:01:30,500") == pytest.approx(90.5)

    def test_hours(self) -> None:
        assert _ts_to_seconds("01:00:00.000") == pytest.approx(3600.0)

    def test_zero(self) -> None:
        assert _ts_to_seconds("00:00:00.000") == pytest.approx(0.0)

    def test_roundtrip_via_seconds_to_ts(self) -> None:
        original = "00:02:15.750"
        secs = _ts_to_seconds(original)
        reconstructed = _seconds_to_ts(secs)
        assert _ts_to_seconds(reconstructed) == pytest.approx(secs, abs=0.001)


# ---------------------------------------------------------------------------
# _parse_vtt
# ---------------------------------------------------------------------------

class TestParseVtt:
    def test_basic_two_segments(self, tmp_path: Path) -> None:
        vtt = _vtt(tmp_path, """\
00:00:01.000 --> 00:00:03.000
Hallo Welt

00:00:04.000 --> 00:00:06.000
Zweiter Satz
""")
        segs = _parse_vtt(vtt)
        assert len(segs) == 2
        assert segs[0]["text"] == "Hallo Welt"
        assert segs[0]["start"] == pytest.approx(1.0)
        assert segs[0]["end"] == pytest.approx(3.0)
        assert segs[1]["text"] == "Zweiter Satz"

    def test_dedup_consecutive_identical(self, tmp_path: Path) -> None:
        """Aufeinanderfolgende identische Texte (YouTube Auto-Captions) werden übersprungen."""
        vtt = _vtt(tmp_path, """\
00:00:01.000 --> 00:00:02.000
Gleicher Text

00:00:02.000 --> 00:00:03.000
Gleicher Text

00:00:03.000 --> 00:00:04.000
Anderer Text
""")
        segs = _parse_vtt(vtt)
        assert len(segs) == 2
        assert segs[0]["text"] == "Gleicher Text"
        assert segs[1]["text"] == "Anderer Text"

    def test_html_entities_decoded(self, tmp_path: Path) -> None:
        vtt = _vtt(tmp_path, """\
00:00:01.000 --> 00:00:03.000
Tom &amp; Jerry &lt;3&gt; It&#39;s &quot;great&quot;
""")
        segs = _parse_vtt(vtt)
        assert len(segs) == 1
        assert segs[0]["text"] == 'Tom & Jerry <3> It\'s "great"'

    def test_vtt_inline_tags_stripped(self, tmp_path: Path) -> None:
        """VTT-interne Tags (<c>, Zeitmarken-Tags) werden entfernt."""
        vtt = _vtt(tmp_path, """\
00:00:01.000 --> 00:00:03.000
<c>Sichtbarer</c> <00:00:01.500><c>Text</c>
""")
        segs = _parse_vtt(vtt)
        assert len(segs) == 1
        assert segs[0]["text"] == "Sichtbarer Text"

    def test_empty_vtt_returns_empty(self, tmp_path: Path) -> None:
        vtt = _vtt(tmp_path, "")
        assert _parse_vtt(vtt) == []

    def test_empty_text_lines_skipped(self, tmp_path: Path) -> None:
        """Blöcke ohne Textinhalt nach Tag-Stripping werden übersprungen."""
        vtt = _vtt(tmp_path, """\
00:00:01.000 --> 00:00:02.000
<c></c>

00:00:03.000 --> 00:00:04.000
Echter Text
""")
        segs = _parse_vtt(vtt)
        assert len(segs) == 1
        assert segs[0]["text"] == "Echter Text"

    def test_non_sequential_identical_not_deduped(self, tmp_path: Path) -> None:
        """Nicht-aufeinanderfolgende identische Texte werden NICHT dedupliziert."""
        vtt = _vtt(tmp_path, """\
00:00:01.000 --> 00:00:02.000
Text A

00:00:02.000 --> 00:00:03.000
Text B

00:00:03.000 --> 00:00:04.000
Text A
""")
        segs = _parse_vtt(vtt)
        assert len(segs) == 3

    def test_comma_decimal_in_timestamp(self, tmp_path: Path) -> None:
        """VTT-Dateien mit Komma als Dezimaltrennzeichen (SRT-Stil) werden akzeptiert."""
        vtt = _vtt(tmp_path, """\
00:00:01,000 --> 00:00:03,000
Komma-Format
""")
        segs = _parse_vtt(vtt)
        assert len(segs) == 1
        assert segs[0]["start"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _write_transcript
# ---------------------------------------------------------------------------

class TestWriteTranscript:
    def test_output_structure(self, tmp_path: Path) -> None:
        import logging
        import src.transcribe as mod
        mod.log = logging.getLogger("test")

        segs = [
            {"start": 1.0, "end": 3.0, "text": "Erster Satz"},
            {"start": 5.0, "end": 7.5, "text": "Zweiter Satz"},
        ]
        out = _write_transcript(tmp_path, segs, source="youtube_subtitles", language="de")
        assert out == tmp_path / "transcript.json"

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["meta"]["source"] == "youtube_subtitles"
        assert data["meta"]["language"] == "de"
        assert data["meta"]["segment_count"] == 2
        assert data["meta"]["duration_seconds"] == pytest.approx(7.5)

        assert data["segments"][0]["text"] == "Erster Satz"
        assert "timestamp" in data["segments"][0]

    def test_whisper_model_in_meta(self, tmp_path: Path) -> None:
        import logging
        import src.transcribe as mod
        mod.log = logging.getLogger("test")

        segs = [{"start": 0.0, "end": 1.0, "text": "Test"}]
        _write_transcript(tmp_path, segs, source="whisper", language="de", model="large")

        data = json.loads((tmp_path / "transcript.json").read_text())
        assert data["meta"]["whisper_model"] == "large"

    def test_empty_segments_duration_zero(self, tmp_path: Path) -> None:
        import logging
        import src.transcribe as mod
        mod.log = logging.getLogger("test")

        _write_transcript(tmp_path, [], source="whisper", language="de")
        data = json.loads((tmp_path / "transcript.json").read_text())
        assert data["meta"]["duration_seconds"] == 0.0
        assert data["segments"] == []
