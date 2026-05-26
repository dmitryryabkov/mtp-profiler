"""CLI for MTP Profiler.

Provides subcommands for each stage of the profiling pipeline:
- parse: Extract telemetry from llama.cpp logs
- analyze: Compute derived metrics
- recommend: Generate MTP setting recommendations
- plot: Generate publication-quality charts

Also supports chained execution:
    mtp-profiler parse analyze recommend plot llama.log
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from mtp_profiler.models.schemas import (
    AnalysisOutput,
    PlotConfig,
    ProfileOutput,
)
from mtp_profiler.parser.log_parser import LlamaCppLogParser, parse_log
from mtp_profiler.analyzer.analyzer import analyze
from mtp_profiler.recommender.recommender import recommend
from mtp_profiler.visualizer.visualizer import plot
from mtp_profiler.system_info.system_info import collect_system_info

app = typer.Typer(
    name="mtp-profiler",
    help="Profile speculative decoding / MTP performance in llama.cpp workloads.",
    add_completion=False,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mtp-profiler")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_json(data: object, path: Path) -> None:
    """Write data to a JSON file with indentation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data.model_dump(mode="json"), f, indent=2, default=str)
    logger.info("Written: %s", path)


def _resolve_path(path_str: str, default_name: str = "output.json") -> Path:
    """Resolve an output path, creating parent dirs if needed."""
    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

@app.command("parse")
def cmd_parse(
    log_file: str = typer.Argument(
        ..., help="Path to llama.cpp server log file"
    ),
    output: str = typer.Option(
        "parsed.json",
        "--output", "-o",
        help="Output JSON file path",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """Parse llama.cpp log file and extract telemetry.

    Reads a llama.cpp server log, extracts timing data, MTP metrics,
    and system information, then writes structured JSON output.
    """
    if verbose:
        logging.getLogger("mtp-profiler").setLevel(logging.DEBUG)

    log_path = Path(log_file)
    if not log_path.exists():
        logger.error("Log file not found: %s", log_path)
        sys.exit(1)

    logger.info("Parsing: %s", log_path)

    parser = LlamaCppLogParser()
    profile = parser.parse_file(log_path)

    out_path = _resolve_path(output, "parsed.json")
    _write_json(profile, out_path)

    # Print summary
    run = profile.runs[0] if profile.runs else None
    if run:
        logger.info(
            "Parsed %d measurements from %d run(s)",
            len(run.measurements),
            len(profile.runs),
        )
        if run.metadata.model:
            logger.info("Model: %s (%s)", run.metadata.model, run.metadata.quantization)
        if run.metadata.system.chip:
            logger.info(
                "System: %s (%d MB RAM)",
                run.metadata.system.chip,
                run.metadata.system.unified_memory_mb,
            )
        if run.warnings:
            logger.info("Warnings: %d", len(run.warnings))

    logger.info("Done. Output: %s", out_path)


@app.command("analyze")
def cmd_analyze(
    input_file: str = typer.Argument(
        ..., help="Path to parsed JSON from the parse stage"
    ),
    output: str = typer.Option(
        "analysis.json",
        "--output", "-o",
        help="Output JSON file path",
    ),
    run_id: str = typer.Option(
        "", "--run-id", "-r",
        help="Specific run ID to analyze (default: first run)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """Analyze parsed telemetry and compute derived metrics.

    Computes throughput statistics, context-length correlations,
    MTP setting comparisons, and stability metrics.
    """
    if verbose:
        logging.getLogger("mtp-profiler").setLevel(logging.DEBUG)

    input_path = Path(input_file)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    logger.info("Loading parsed data from: %s", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        profile_dict = json.load(f)

    profile = ProfileOutput.model_validate(profile_dict)

    run_id_param = run_id if run_id else None
    analysis = analyze(profile, run_id_param)

    out_path = _resolve_path(output, "analysis.json")
    _write_json(analysis, out_path)

    # Print summary
    metrics = analysis.metrics
    logger.info("Analysis complete for run: %s", analysis.run_id)
    if metrics.avg_generation_tps:
        logger.info("Avg generation throughput: %.2f t/s", metrics.avg_generation_tps)
    if metrics.avg_acceptance_rate is not None:
        logger.info("Avg acceptance rate: %.1f%%", metrics.avg_acceptance_rate * 100)
    if metrics.context_tps_correlation is not None:
        logger.info("Context-TPS correlation: %.4f", metrics.context_tps_correlation)
    logger.info("MTP settings compared: %d", len(analysis.mtp_setting_comparisons))
    logger.info("Done. Output: %s", out_path)


@app.command("recommend")
def cmd_recommend(
    input_file: str = typer.Argument(
        ..., help="Path to analysis JSON from the analyze stage"
    ),
    output: str = typer.Option(
        "recommendation.json",
        "--output", "-o",
        help="Output JSON file path",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """Generate MTP setting recommendations.

    Evaluates analyzed data to determine the optimal MTP draft-token
    setting based on throughput, stability, and long-context efficiency.
    """
    if verbose:
        logging.getLogger("mtp-profiler").setLevel(logging.DEBUG)

    input_path = Path(input_file)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    logger.info("Loading analysis from: %s", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        analysis_dict = json.load(f)

    analysis = AnalysisOutput.model_validate(analysis_dict)

    # Also load the parsed data for context
    parsed_path = input_path.parent / "parsed.json"
    profile: Optional[ProfileOutput] = None
    if parsed_path.exists():
        with open(parsed_path, "r", encoding="utf-8") as f:
            profile = ProfileOutput.model_validate(json.load(f))

    rec_output = recommend(profile, analysis)

    out_path = _resolve_path(output, "recommendation.json")
    _write_json(rec_output, out_path)

    # Print recommendation
    print(f"\n{'='*60}")
    print(f"  MTP Profiler - Recommendation")
    print(f"{'='*60}")
    print(rec_output.summary_text)
    print(f"{'='*60}\n")

    logger.info("Done. Output: %s", out_path)


@app.command("plot")
def cmd_plot(
    input_file: str = typer.Argument(
        ..., help="Path to analysis JSON from the analyze stage"
    ),
    output_dir: str = typer.Option(
        "charts",
        "--output-dir", "-d",
        help="Directory to save charts",
    ),
    run_id: str = typer.Option(
        "", "--run-id", "-r",
        help="Specific run ID to plot (default: first run)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """Generate publication-quality charts.

    Creates charts showing:
    - Generation throughput vs context length
    - Draft acceptance rate trends
    - MTP setting comparison (uplift vs baseline)
    - Throughput stability (boxplot)
    """
    if verbose:
        logging.getLogger("mtp-profiler").setLevel(logging.DEBUG)

    input_path = Path(input_file)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    logger.info("Loading analysis from: %s", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        analysis_dict = json.load(f)

    analysis = AnalysisOutput.model_validate(analysis_dict)

    # Load parsed data
    parsed_path = input_path.parent / "parsed.json"
    if not parsed_path.exists():
        logger.error("parsed.json not found in %s", input_path.parent)
        sys.exit(1)

    with open(parsed_path, "r", encoding="utf-8") as f:
        profile = ProfileOutput.model_validate(json.load(f))

    run_id_param = run_id if run_id else None
    config = PlotConfig()
    chart_paths = plot(profile, analysis, output_dir, config, run_id_param)

    if chart_paths:
        print(f"\nGenerated {len(chart_paths)} chart(s):")
        for p in chart_paths:
            print(f"  - {p}")
        print()

    logger.info("Done. Charts saved to: %s", output_dir)


# ---------------------------------------------------------------------------
# Chained convenience command
# ---------------------------------------------------------------------------

@app.command("profile")
def cmd_profile(
    log_file: str = typer.Argument(
        ..., help="Path to llama.cpp server log file"
    ),
    output_dir: str = typer.Option(
        "mtp-output",
        "--output-dir", "-d",
        help="Output directory for all artifacts",
    ),
    run_id: str = typer.Option(
        "", "--run-id", "-r",
        help="Specific run ID to process (default: first run)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """Full profiling pipeline: parse -> analyze -> recommend -> plot.

    Runs all stages in sequence on a single log file, producing:
    - parsed.json: Raw extracted telemetry
    - analysis.json: Computed metrics
    - recommendation.json: Optimal MTP setting
    - charts/: Generated visualizations
    """
    if verbose:
        logging.getLogger("mtp-profiler").setLevel(logging.DEBUG)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    charts_dir = output_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    parsed_path = output_path / "parsed.json"
    analysis_path = output_path / "analysis.json"
    recommendation_path = output_path / "recommendation.json"

    log_path = Path(log_file)
    if not log_path.exists():
        logger.error("Log file not found: %s", log_path)
        sys.exit(1)

    run_id_param = run_id if run_id else None

    # Stage 1: Parse
    logger.info("=" * 60)
    logger.info("Stage 1/4: Parsing log file")
    logger.info("=" * 60)
    parser = LlamaCppLogParser()
    profile = parser.parse_file(log_path)
    _write_json(profile, parsed_path)
    if profile.runs:
        run = profile.runs[0]
        logger.info("  Extracted %d measurements", len(run.measurements))

    # Stage 2: Analyze
    logger.info("=" * 60)
    logger.info("Stage 2/4: Analyzing telemetry")
    logger.info("=" * 60)
    analysis = analyze(profile, run_id_param)
    _write_json(analysis, analysis_path)
    metrics = analysis.metrics
    if metrics.avg_generation_tps:
        logger.info("  Avg generation throughput: %.2f t/s", metrics.avg_generation_tps)

    # Stage 3: Recommend
    logger.info("=" * 60)
    logger.info("Stage 3/4: Generating recommendations")
    logger.info("=" * 60)
    rec_output = recommend(profile, analysis)
    _write_json(rec_output, recommendation_path)
    print(f"\n{'='*60}")
    print(f"  MTP Profiler - Recommendation")
    print(f"{'='*60}")
    print(rec_output.summary_text)
    print(f"{'='*60}\n")

    # Stage 4: Plot
    logger.info("=" * 60)
    logger.info("Stage 4/4: Generating charts")
    logger.info("=" * 60)
    config = PlotConfig()
    chart_paths = plot(profile, analysis, charts_dir, config, run_id_param)
    if chart_paths:
        logger.info("  Generated %d chart(s)", len(chart_paths))

    logger.info("=" * 60)
    logger.info("Done! All artifacts in: %s", output_path)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# System info command
# ---------------------------------------------------------------------------

@app.command("sysinfo")
def cmd_sysinfo() -> None:
    """Display system information relevant to MTP profiling."""
    info = collect_system_info()

    print(f"\n{'='*60}")
    print(f"  System Information")
    print(f"{'='*60}")
    print(f"  Apple Silicon : {'Yes' if info.is_apple_silicon else 'No'}")
    print(f"  Chip          : {info.chip or 'Unknown'}")
    print(f"  Chip Type     : {info.chip_type or 'Unknown'}")
    print(f"  Memory        : {info.unified_memory_mb} MB")
    print(f"  macOS Version : {info.macos_version or 'Unknown'}")
    print(f"  CPU Threads   : {info.cpu_threads}")
    print(f"  Total Threads : {info.cpu_total_threads}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the mtp-profiler CLI."""
    app()


if __name__ == "__main__":
    main()
