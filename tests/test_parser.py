"""Tests for the llama.cpp log parser."""

import json
import tempfile
from pathlib import Path

import pytest

from mtp_profiler.models.schemas import ProfileOutput
from mtp_profiler.parser.log_parser import (
    LlamaCppLogParser,
    _strip_ansi,
    _parse_timestamp,
    parse_log,
)
from tests.fixtures import (
    SAMPLE_LOG,
    SAMPLE_LOG_EMPTY,
    SAMPLE_LOG_GARBAGE,
    SAMPLE_LOG_MULTI_RUN,
    SAMPLE_LOG_NO_ANSI,
    SAMPLE_LOG_TRUNCATED,
    PARSED_JSON_FIXTURE,
)


class TestStripAnsi:
    """Tests for ANSI code stripping."""

    def test_strips_ansi_codes(self):
        result = _strip_ansi("\x1b[34mhello\x1b[0m")
        assert result == "hello"

    def test_strips_multiple_ansi_codes(self):
        result = _strip_ansi("\x1b[34m0.00.052.030\x1b[0m \x1b[32mI \x1b[0mtest")
        assert result == "0.00.052.030 I test"

    def test_no_ansi_unchanged(self):
        result = _strip_ansi("plain text")
        assert result == "plain text"

    def test_empty_string(self):
        result = _strip_ansi("")
        assert result == ""


class TestParseTimestamp:
    """Tests for timestamp parsing."""

    def test_valid_timestamp(self):
        result = _parse_timestamp("10.54.639.911")
        assert result == pytest.approx(654.639911)

    def test_zero_timestamp(self):
        result = _parse_timestamp("00.00.000.000")
        assert result == 0.0

    def test_invalid_format(self):
        result = _parse_timestamp("not-a-timestamp")
        assert result is None

    def test_partial_timestamp(self):
        result = _parse_timestamp("10.54")
        assert result is None


class TestParserBasic:
    """Basic parser functionality tests."""

    def test_parse_sample_log(self):
        """Test parsing the sample log with ANSI codes."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG)
            f.flush()
            profile = parse_log(f.name)

        assert len(profile.runs) == 1
        run = profile.runs[0]
        assert len(run.measurements) >= 2

    def test_parse_no_ansi_log(self):
        """Test parsing a log without ANSI codes."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG_NO_ANSI)
            f.flush()
            profile = parse_log(f.name)

        assert len(profile.runs) == 1
        run = profile.runs[0]
        assert len(run.measurements) >= 1

    def test_parse_empty_log(self):
        """Test parsing an empty log file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG_EMPTY)
            f.flush()
            profile = parse_log(f.name)

        assert len(profile.runs) == 1
        assert len(profile.runs[0].measurements) == 0

    def test_parse_nonexistent_file(self):
        """Test that parsing a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_log("/nonexistent/path/to/log.log")


class TestParserExtraction:
    """Tests for specific field extraction."""

    def test_model_extraction(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG)
            f.flush()
            profile = parse_log(f.name)

        run = profile.runs[0]
        assert run.metadata.model == "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
        assert run.metadata.quantization == "Q4_K_XL"

    def test_system_info_extraction(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG)
            f.flush()
            profile = parse_log(f.name)

        run = profile.runs[0]
        assert run.metadata.system.chip == "Apple M3 Pro"
        assert run.metadata.system.chip_type == "M3"
        assert run.metadata.system.unified_memory_mb == 28753

    def test_mtp_config_extraction(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG)
            f.flush()
            profile = parse_log(f.name)

        run = profile.runs[0]
        assert run.metadata.mtp_config["n_max"] == 2
        assert run.metadata.mtp_config["n_min"] == 0
        assert run.metadata.mtp_config["p_min"] == 0.7

    def test_threads_extraction(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG)
            f.flush()
            profile = parse_log(f.name)

        run = profile.runs[0]
        assert run.metadata.system.cpu_threads == 8
        assert run.metadata.system.cpu_total_threads == 11

    def test_measurement_fields(self):
        """Test that all key measurement fields are extracted."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG)
            f.flush()
            profile = parse_log(f.name)

        run = profile.runs[0]
        # First measurement
        m0 = run.measurements[0]
        assert m0.n_tokens == 4430
        assert m0.prompt_tokens_per_second == 386.48
        assert m0.generation_tokens_per_second == 30.81
        assert m0.draft_acceptance_rate == 0.89809
        assert m0.n_drafts_generated == 785
        assert m0.n_drafts_accepted == 705
        assert m0.prompt_eval_time_ms == 8119.49
        assert m0.eval_time_ms == 41904.01
        assert m0.total_time_ms == 50023.50

    def test_multiple_measurements(self):
        """Test that multiple measurements from different tasks are extracted."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG)
            f.flush()
            profile = parse_log(f.name)

        run = profile.runs[0]
        # Should have 2 measurements (task 0 and task 2)
        assert len(run.measurements) == 2


class TestParserResilience:
    """Tests for parser resilience to malformed input."""

    def test_garbage_lines_skipped(self):
        """Test that garbage lines don't crash the parser."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG_GARBAGE)
            f.flush()
            profile = parse_log(f.name)

        assert len(profile.runs) == 1
        run = profile.runs[0]
        # Should still extract valid measurements from garbage
        assert len(run.measurements) >= 1

    def test_truncated_log(self):
        """Test that truncated logs don't crash."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG_TRUNCATED)
            f.flush()
            profile = parse_log(f.name)

        assert len(profile.runs) == 1
        # May or may not extract measurements from truncated lines
        # The important thing is it doesn't crash

    def test_utf8_errors_handled(self):
        """Test that UTF-8 decode errors don't crash."""
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".log", delete=False
        ) as f:
            f.write(b"\xff\xfe invalid utf8\n")
            f.write(
                b"\x1b[34m10.54.639.911\x1b[0m "
                b"\x1b[32mI \x1b[0mslot print_timing: "
                b"id  0 | task 0 | "
                b"eval time =   41904.01 ms /  1291 tokens "
                b"(   32.46 ms per token,    30.81 tokens per second)\n"
            )
            f.flush()
            profile = parse_log(f.name)

        assert len(profile.runs) == 1
        run = profile.runs[0]
        assert len(run.measurements) >= 1


class TestParserAgainstAwkReference:
    """Validate parser output against the awk reference implementation."""

    def test_awk_equivalent_extraction(self):
        """
        The awk command extracts: n_tokens, prompt_tps, gen_tps, acceptance_pct.
        Verify our parser extracts the same fields from the same log lines.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG)
            f.flush()
            profile = parse_log(f.name)

        run = profile.runs[0]
        # Find the measurement with draft acceptance (like the awk filter)
        measurements_with_acceptance = [
            m for m in run.measurements if m.draft_acceptance_rate is not None
        ]

        assert len(measurements_with_acceptance) >= 1

        # Verify each has the expected fields
        for m in measurements_with_acceptance:
            assert m.n_tokens is not None or m.n_decoded is not None
            assert m.prompt_tokens_per_second is not None
            assert m.generation_tokens_per_second is not None
            assert m.draft_acceptance_rate is not None

    def test_acceptance_rate_values(self):
        """Verify acceptance rates are in valid range [0, 1]."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG)
            f.flush()
            profile = parse_log(f.name)

        for m in profile.runs[0].measurements:
            if m.draft_acceptance_rate is not None:
                assert 0 <= m.draft_acceptance_rate <= 1

    def test_throughput_values_positive(self):
        """Verify throughput values are positive when present."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG)
            f.flush()
            profile = parse_log(f.name)

        for m in profile.runs[0].measurements:
            if m.generation_tokens_per_second is not None:
                assert m.generation_tokens_per_second > 0
            if m.prompt_tokens_per_second is not None:
                assert m.prompt_tokens_per_second > 0


class TestProfileOutputSchema:
    """Tests for the ProfileOutput data model."""

    def test_parse_to_model(self):
        """Test that parsed output validates against the schema."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG)
            f.flush()
            profile = parse_log(f.name)

        # Should validate without error
        validated = ProfileOutput.model_validate(profile.model_dump(mode="json"))
        assert len(validated.runs) == 1

    def test_json_roundtrip(self):
        """Test that output can be serialized and deserialized."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG)
            f.flush()
            profile = parse_log(f.name)

        json_str = json.dumps(profile.model_dump(mode="json"), default=str)
        loaded = ProfileOutput.model_validate(json.loads(json_str))
        assert len(loaded.runs) == len(profile.runs)


class TestMultiRun:
    """Tests for multi-server-instance log parsing."""

    def test_multi_run_detection(self):
        """Test that multiple server instances create separate runs."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG_MULTI_RUN)
            f.flush()
            profile = parse_log(f.name)

        assert len(profile.runs) == 2
        assert profile.runs[0].id == "run_1"
        assert profile.runs[1].id == "run_2"

    def test_multi_run_mtp_configs(self):
        """Test that each run has its own MTP config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG_MULTI_RUN)
            f.flush()
            profile = parse_log(f.name)

        assert profile.runs[0].metadata.mtp_config["n_max"] == 1
        assert profile.runs[1].metadata.mtp_config["n_max"] == 2

    def test_multi_run_measurements(self):
        """Test that measurements are correctly separated by run."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG_MULTI_RUN)
            f.flush()
            profile = parse_log(f.name)

        assert len(profile.runs[0].measurements) == 1
        assert len(profile.runs[1].measurements) == 1

    def test_multi_run_system_info_preserved(self):
        """Test that system info is preserved across runs."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG_MULTI_RUN)
            f.flush()
            profile = parse_log(f.name)

        for run in profile.runs:
            assert run.metadata.system.chip == "Apple M3 Pro"
            assert run.metadata.system.unified_memory_mb == 28753
