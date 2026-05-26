"""llama.cpp log parser for MTP Profiler.

Parses real-world llama.cpp server logs to extract telemetry
about prompt processing, generation throughput, and MTP
(speculative decoding) performance metrics.

Design principles:
- Resilient to log variation (ANSI codes, formatting changes)
- Never crashes on malformed input
- Skips unparseable lines with warnings
- Preserves all recoverable telemetry
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mtp_profiler.models.schemas import (
    Measurement,
    ProfileOutput,
    Run,
    RunMetadata,
    SystemInfo,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex patterns for llama.cpp log lines
# ---------------------------------------------------------------------------

# Strip ANSI escape codes
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Timestamp pattern: MM.SSS.MMM (e.g. 42.09.594.216)
_TIMESTAMP_RE = re.compile(r"(\d{2}\.\d{2}\.\d{3}\.\d{3})")

# Slot ID and task ID
_SLOT_ID_RE = re.compile(r"id\s+(\d+)")
_TASK_ID_RE = re.compile(r"task\s+(\d+)")

# Model path (from load_model lines)
_MODEL_PATH_RE = re.compile(
    r"load_model:\s*loading model\s*'([^']+\.gguf)'"
)

# Quantization from filename (e.g. Q4_K_XL)
_QUANT_RE = re.compile(r"Q[0-9]_[A-Z_]+")

# llama.cpp version (from various version/info lines)
_VERSION_RE = re.compile(r"llama\.cpp\s+([0-9a-f.]+)")

# Device info: Apple M3 Pro (28753 MiB, 28753 MiB free)
_DEVICE_RE = re.compile(
    r"MTL\d+\s*:\s*(Apple\s+[A-Za-z0-9\s]+?)\s*\((\d+)\s*Mib",
    re.IGNORECASE,
)

# System info: n_threads = X (n_threads_batch = Y) / Z
_THREADS_RE = re.compile(r"n_threads\s*=\s*(\d+)\s*\(n_threads_batch\s*=\s*(\d+)\)\s*/\s*(\d+)")

# MTP config: n_max=N, n_min=M, p_min=P
_MTP_CONFIG_RE = re.compile(
    r"n_max=(\d+),\s*n_min=(\d+),\s*p_min=([0-9.]+)"
)

# Prompt processing (progressive, during processing)
# e.g. "prompt processing, n_tokens =  2110, progress = 0.67, t =   5.54 s / 381.03 tokens per second"
_PROMPT_PROCESSING_RE = re.compile(
    r"prompt processing,\s*n_tokens\s*=\s*(\d+),\s*progress\s*=\s*([0-9.]+),\s*t\s*=\s*([0-9.]+)\s*s\s*/\s*([0-9.]+)\s*tokens per second"
)

# Prompt eval time (final summary)
# e.g. "prompt eval time =    2416.79 ms /   400 tokens (    6.04 ms per token,   165.51 tokens per second)"
_PROMPT_EVAL_RE = re.compile(
    r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\)"
)

# Eval time (generation, final summary)
# e.g. "       eval time =    8749.26 ms /   234 tokens (   37.39 ms per token,    26.75 tokens per second)"
_EVAL_RE = re.compile(
    r"eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\)"
)

# Total time
# e.g. "      total time =   11166.05 ms /   634 tokens"
_TOTAL_TIME_RE = re.compile(
    r"total time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens"
)

# Decoded tokens with generation speed
# e.g. "n_decoded =    101, tg =  26.67 t/s"
_DECODED_RE = re.compile(
    r"n_decoded\s*=\s*(\d+),\s*tg\s*=\s*([0-9.]+)\s*t/s"
)

# Draft acceptance
# e.g. "draft acceptance = 0.98261 (  113 accepted /   115 generated)"
_DRAFT_ACCEPTANCE_RE = re.compile(
    r"draft acceptance\s*=\s*([0-9.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\)"
)

# MTP statistics line
# e.g. "draft-mtp: #calls(b,g,a) =    1    157    130, #gen drafts =    130, #acc drafts =   120, #gen tokens =    130, #acc tokens =   120, dur(b,g,a) = 0.001, 2758.959, 0.185 ms"
_STATS_MTP_RE = re.compile(
    r"draft-mtp:\s*#calls\(b,g,a\)\s*=\s*(\d+)\s+(\d+)\s+(\d+),\s*"
    r"#gen drafts\s*=\s*(\d+),\s*#acc drafts\s*=\s*(\d+),\s*"
    r"#gen tokens\s*=\s*(\d+),\s*#acc tokens\s*=\s*(\d+),\s*"
    r"dur\(b,g,a\)\s*=\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\s+ms"
)

# Context checkpoint (only n_tokens)
# e.g. "created context checkpoint 1 of 16 (pos_min = 2109, pos_max = 2109, n_tokens = 2110, size = 66.975 MiB)"
_CHECKPOINT_RE = re.compile(
    r"created context checkpoint\s+\d+\s+of\s+\d+\s+\(.*n_tokens\s*=\s*(\d+)"
)

# Release / stop processing
# e.g. "stop processing: n_tokens = 34510, truncated = 0"
_RELEASE_RE = re.compile(
    r"stop processing:\s*n_tokens\s*=\s*(\d+),\s*truncated\s*=\s*(\d+)"
)


@dataclass
class _ParseContext:
    """Internal state for incremental log parsing."""

    _current_slot: Optional[int] = None
    _current_task: Optional[int] = None
    _measurement_task: Optional[int] = None
    current_measurement: Optional[Measurement] = None
    measurements: list[Measurement] = field(default_factory=list)
    metadata: RunMetadata = field(default_factory=RunMetadata)
    system_info: SystemInfo = field(default_factory=SystemInfo)
    warnings: list[str] = field(default_factory=list)
    first_timestamp: Optional[float] = None
    last_timestamp: Optional[float] = None
    # Track runs for multi-server-instance logs
    runs: list[Run] = field(default_factory=list)
    _run_counter: int = 0
    _has_measurements: bool = False


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from a log line."""
    return _ANSI_RE.sub("", text)


def _parse_timestamp(ts_str: str) -> Optional[float]:
    """Parse a llama.cpp timestamp (MM.SSS.MMM) to epoch-like float seconds."""
    try:
        parts = ts_str.split(".")
        if len(parts) != 4:
            return None
        minutes = int(parts[0])
        seconds = int(parts[1])
        millis = int(parts[2])
        micros = int(parts[3])
        return minutes * 60 + seconds + millis / 1000 + micros / 1_000_000
    except (ValueError, IndexError):
        return None


def _extract_model_info(line: str) -> tuple[str, str]:
    """Extract model path and quantization from a log line."""
    m = _MODEL_PATH_RE.search(line)
    if not m:
        return "", ""
    path = m.group(1)
    filename = Path(path).name
    quant = _QUANT_RE.search(filename)
    quant_str = quant.group(0) if quant else ""
    return filename, quant_str


class LlamaCppLogParser:
    """Robust parser for llama.cpp server logs.

    Parses logs incrementally, building up a structured representation
    of inference runs and their telemetry measurements.
    """

    def __init__(self):
        self._ctx = _ParseContext()

    def parse_file(self, path: Path | str) -> ProfileOutput:
        """Parse a llama.cpp log file and return structured output."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {path}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line_num, raw_line in enumerate(f, 1):
                try:
                    self._parse_line(raw_line, line_num)
                except Exception as e:
                    self._ctx.warnings.append(
                        f"Line {line_num}: unexpected error: {e}"
                    )
                    logger.warning("Error on line %d: %s", line_num, e)

        # Finalize any open measurement and current run
        if self._ctx.current_measurement is not None or self._ctx._has_measurements:
            self._finalize_current_run()
        elif self._ctx.current_measurement is not None:
            m = self._ctx.current_measurement
            has_data = any([
                m.n_tokens is not None,
                m.n_decoded is not None,
                m.generation_tokens_per_second is not None,
                m.prompt_tokens_per_second is not None,
                m.draft_acceptance_rate is not None,
                m.eval_time_ms is not None,
                m.prompt_eval_time_ms is not None,
                m.total_time_ms is not None,
                m.n_drafts_generated is not None,
                m.n_drafts_accepted is not None,
            ])
            if has_data:
                self._ctx.measurements.append(m)
            else:
                self._ctx.warnings.append(
                    f"Discarded empty measurement"
                )
            self._ctx.current_measurement = None

        # If no runs were created (no measurements at all), create empty run
        if not self._ctx.runs:
            self._ctx.metadata.system = self._ctx.system_info
            run = Run(
                id="run_1",
                metadata=self._ctx.metadata,
                measurements=self._ctx.measurements,
                warnings=self._ctx.warnings,
            )
            self._ctx.runs.append(run)

        return ProfileOutput(
            runs=self._ctx.runs,
            parse_warnings=self._ctx.warnings,
        )

    def _parse_line(self, raw_line: str, line_num: int) -> None:
        """Parse a single log line and update internal state."""
        line = _strip_ansi(raw_line).strip()
        if not line:
            return

        # Extract timestamp
        ts_match = _TIMESTAMP_RE.search(line)
        ts_seconds = None
        if ts_match:
            ts_seconds = _parse_timestamp(ts_match.group(1))
            if ts_seconds is not None:
                if self._ctx.first_timestamp is None:
                    self._ctx.first_timestamp = ts_seconds
                self._ctx.last_timestamp = ts_seconds

        # Extract slot/task IDs from various line formats
        slot_match = _SLOT_ID_RE.search(line)
        task_match = _TASK_ID_RE.search(line)
        if slot_match:
            self._ctx._current_slot = int(slot_match.group(1))
        if task_match:
            self._ctx._current_task = int(task_match.group(1))

        # Check for task change: compare current_task with the task that started this measurement
        task_changed_for_current = (
            self._ctx.current_measurement is not None
            and self._ctx._measurement_task is not None
            and self._ctx._current_task is not None
            and self._ctx._measurement_task != self._ctx._current_task
        )

        # System / hardware info
        self._parse_system_info(line)

        # Model info
        model_name, quant = _extract_model_info(line)
        if model_name and not self._ctx.metadata.model:
            self._ctx.metadata.model = model_name
        if quant and not self._ctx.metadata.quantization:
            self._ctx.metadata.quantization = quant

        # MTP config
        mtp_match = _MTP_CONFIG_RE.search(line)
        if mtp_match:
            self._ctx.metadata.mtp_config["n_max"] = int(mtp_match.group(1))
            self._ctx.metadata.mtp_config["n_min"] = int(mtp_match.group(2))
            self._ctx.metadata.mtp_config["p_min"] = float(mtp_match.group(3))

        # Ensure we have a current measurement to populate
        # If task changed, finalize the current measurement and start a new one
        if (
            self._ctx.current_measurement is not None
            and self._ctx._current_task is not None
            and self._ctx._measurement_task is not None
            and self._ctx._measurement_task != self._ctx._current_task
        ):
            self._ctx.measurements.append(self._ctx.current_measurement)
            self._ctx.current_measurement = Measurement()
            self._ctx._measurement_task = self._ctx._current_task
        elif self._ctx.current_measurement is None:
            self._ctx.current_measurement = Measurement()
            self._ctx._measurement_task = self._ctx._current_task
        elif (
            self._ctx.current_measurement is not None
            and self._ctx._current_task is not None
            and self._ctx._measurement_task is None
        ):
            self._ctx._measurement_task = self._ctx._current_task

        m = self._ctx.current_measurement

       # Track that we have measurements
        if m.n_tokens is not None or m.n_decoded is not None:
            self._ctx._has_measurements = True

        # Prompt processing (progressive)
        pp_match = _PROMPT_PROCESSING_RE.search(line)
        if pp_match:
            m.prompt_eval_time_ms = float(pp_match.group(3)) * 1000
            m.prompt_tokens_per_second = float(pp_match.group(4))

        # Prompt eval time (final)
        pe_match = _PROMPT_EVAL_RE.search(line)
        if pe_match:
            m.prompt_eval_time_ms = float(pe_match.group(1))
            m.n_tokens = int(pe_match.group(2))
            m.prompt_tokens_per_second = float(pe_match.group(4))

        # Eval time (generation)
        ev_match = _EVAL_RE.search(line)
        if ev_match:
            m.eval_time_ms = float(ev_match.group(1))
            m.n_decoded = int(ev_match.group(2))
            m.generation_tokens_per_second = float(ev_match.group(4))

        # Total time
        tt_match = _TOTAL_TIME_RE.search(line)
        if tt_match:
            m.total_time_ms = float(tt_match.group(1))
            if m.n_tokens is None:
                m.n_tokens = int(tt_match.group(2))

        # Decoded tokens with generation speed
        dec_match = _DECODED_RE.search(line)
        if dec_match:
            decoded_val = int(dec_match.group(1))
            if m.n_decoded is None or decoded_val > m.n_decoded:
                m.n_decoded = decoded_val
            m.generation_tokens_per_second = float(dec_match.group(2))

        # Draft acceptance
        da_match = _DRAFT_ACCEPTANCE_RE.search(line)
        if da_match:
            m.draft_acceptance_rate = float(da_match.group(1))
            m.n_drafts_accepted = int(da_match.group(2))
            m.n_drafts_generated = int(da_match.group(3))

        # MTP statistics
        stats_match = _STATS_MTP_RE.search(line)
        if stats_match:
            calls = int(stats_match.group(1))
            gen_drafts_calls = int(stats_match.group(2))
            acc_drafts_calls = int(stats_match.group(3))
            gen_drafts = int(stats_match.group(4))
            acc_drafts = int(stats_match.group(5))
            gen_tokens = int(stats_match.group(6))
            acc_tokens = int(stats_match.group(7))
            dur_batch = float(stats_match.group(8))
            dur_gen = float(stats_match.group(9))
            dur_acc = float(stats_match.group(10))

            m.mtp_calls = calls
            m.mtp_gen_drafts = gen_drafts
            m.mtp_acc_drafts = acc_drafts
            m.mtp_gen_tokens = gen_tokens
            m.mtp_acc_tokens = acc_tokens
            m.mtp_dur_batch = dur_batch
            m.mtp_dur_gen = dur_gen
            m.mtp_dur_acc = dur_acc

            if m.n_drafts_generated is None:
                m.n_drafts_generated = gen_tokens
            if m.n_drafts_accepted is None:
                m.n_drafts_accepted = acc_tokens

        # Context checkpoint (only n_tokens, not size)
        ck_match = _CHECKPOINT_RE.search(line)
        if ck_match:
            m.n_tokens = int(ck_match.group(1))

        # Release / stop processing
        rel_match = _RELEASE_RE.search(line)
        if rel_match:
            prev_n_tokens = m.n_tokens
            m.n_tokens = int(rel_match.group(1))
            m.truncated = int(rel_match.group(2))
            # If context length grew significantly, finalize and start new measurement
            # Also finalize if task_id changed (different inference session)
            context_grew = (
                prev_n_tokens is not None
                and m.n_tokens > prev_n_tokens * 1.5
            )
            if context_grew or task_changed_for_current:
                self._ctx.measurements.append(self._ctx.current_measurement)
                self._ctx.current_measurement = Measurement()

    def _finalize_current_run(self) -> None:
        """Finalize the current run and add it to the runs list."""
        # Only create a run if we have measurements
        if not self._ctx.measurements and self._ctx.current_measurement is None:
            return

        # Finalize any open measurement
        if self._ctx.current_measurement is not None:
            m = self._ctx.current_measurement
            has_data = any([
                m.n_tokens is not None,
                m.n_decoded is not None,
                m.generation_tokens_per_second is not None,
                m.prompt_tokens_per_second is not None,
                m.draft_acceptance_rate is not None,
                m.eval_time_ms is not None,
                m.prompt_eval_time_ms is not None,
                m.total_time_ms is not None,
                m.n_drafts_generated is not None,
                m.n_drafts_accepted is not None,
            ])
            if has_data:
                self._ctx.measurements.append(m)
            else:
                self._ctx.warnings.append("Discarded empty measurement")
            self._ctx.current_measurement = None

        # Only create run if we have measurements after finalization
        if not self._ctx.measurements:
            return

        # Transfer system info and create the run
        self._ctx.metadata.system = self._ctx.system_info
        self._ctx._run_counter += 1
        run = Run(
            id=f"run_{self._ctx._run_counter}",
            metadata=self._ctx.metadata,
            measurements=self._ctx.measurements,
            warnings=list(self._ctx.warnings),
        )
        self._ctx.runs.append(run)

        # Reset for new run (keep system_info)
        self._ctx.metadata = RunMetadata()
        self._ctx.measurements = []
        self._ctx.warnings = []
        self._ctx._measurement_task = None

    def _is_server_restart(self, line: str) -> bool:
        """Check if this line indicates a new server instance startup."""
        # Device info line (MTL0) indicates server startup
        if _DEVICE_RE.search(line):
            return True
        # Log verbosity line indicates server startup
        if "log_info: verbosity" in line:
            return True
        return False

    def _parse_system_info(self, line: str) -> None:
        """Extract system/hardware info from log lines."""
        # Check for server restart before processing system info
        if self._ctx._has_measurements and self._is_server_restart(line):
            self._finalize_current_run()

        # Device info
        dev_match = _DEVICE_RE.search(line)
        if dev_match:
            chip_name = dev_match.group(1).strip()
            mem_mb = int(dev_match.group(2))
            self._ctx.system_info.chip = chip_name
            self._ctx.system_info.unified_memory_mb = mem_mb
            # Extract chip type (M1, M2, M3, etc.)
            chip_type_match = re.search(r"(M\d+)", chip_name)
            if chip_type_match:
                self._ctx.system_info.chip_type = chip_type_match.group(1)

        # Threads
        th_match = _THREADS_RE.search(line)
        if th_match:
            self._ctx.system_info.cpu_threads = int(th_match.group(1))
            self._ctx.system_info.cpu_total_threads = int(th_match.group(3))


def parse_log(path: Path | str) -> ProfileOutput:
    """Convenience function to parse a llama.cpp log file."""
    parser = LlamaCppLogParser()
    return parser.parse_file(path)
