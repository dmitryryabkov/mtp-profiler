"""Pydantic data models for MTP Profiler."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class SystemInfo(BaseModel):
    """Hardware and OS information for Apple Silicon systems."""

    chip: str = ""
    chip_type: str = ""  # M1, M2, M3, etc.
    unified_memory_mb: int = 0
    macos_version: str = ""
    cpu_threads: int = 0
    cpu_total_threads: int = 0


class RunMetadata(BaseModel):
    """Top-level metadata for a profiling run."""

    model: str = ""
    quantization: str = ""
    llama_cpp_version: str = ""
    system: SystemInfo = Field(default_factory=SystemInfo)
    mtp_config: dict[str, Any] = Field(default_factory=dict)


class Measurement(BaseModel):
    """A single telemetry measurement from a llama.cpp inference session."""

    # Token counts
    n_decoded: Optional[int] = None
    n_tokens: Optional[int] = None
    n_drafts_generated: Optional[int] = None
    n_drafts_accepted: Optional[int] = None
    truncated: Optional[int] = None

    # Timing (in milliseconds)
    prompt_eval_time_ms: Optional[float] = None
    eval_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None

    # Throughput (tokens per second)
    prompt_tokens_per_second: Optional[float] = None
    generation_tokens_per_second: Optional[float] = None

    # MTP / speculative decoding
    draft_acceptance_rate: Optional[float] = None

    # MTP internal stats (from statistics draft-mtp line)
    mtp_calls: Optional[int] = None
    mtp_gen_drafts: Optional[int] = None
    mtp_acc_drafts: Optional[int] = None
    mtp_gen_tokens: Optional[int] = None
    mtp_acc_tokens: Optional[int] = None
    mtp_dur_batch: Optional[float] = None
    mtp_dur_gen: Optional[float] = None
    mtp_dur_acc: Optional[float] = None


class Run(BaseModel):
    """A complete profiling run with metadata and measurements."""

    id: str = ""
    metadata: RunMetadata = Field(default_factory=RunMetadata)
    measurements: list[Measurement] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ProfileOutput(BaseModel):
    """Top-level output of the parse stage."""

    runs: list[Run] = Field(default_factory=list)
    parse_warnings: list[str] = Field(default_factory=list)


class AnalysisMetrics(BaseModel):
    """Derived metrics from the analyze stage."""

    # Throughput statistics
    avg_generation_tps: Optional[float] = None
    std_generation_tps: Optional[float] = None
    min_generation_tps: Optional[float] = None
    max_generation_tps: Optional[float] = None
    median_generation_tps: Optional[float] = None
    p10_generation_tps: Optional[float] = None
    p90_generation_tps: Optional[float] = None

    avg_prompt_tps: Optional[float] = None
    std_prompt_tps: Optional[float] = None

    avg_acceptance_rate: Optional[float] = None
    std_acceptance_rate: Optional[float] = None

    # Context-length correlation
    context_tps_correlation: Optional[float] = None
    context_degradation_rate: Optional[float] = None  # tps per 1000 tokens of context

    # MTP-specific
    avg_drafts_per_generation: Optional[float] = None
    avg_accepted_per_generation: Optional[float] = None

    # Stability
    tps_variance: Optional[float] = None
    tps_cv: Optional[float] = None  # coefficient of variation

    # Long-context behavior
    short_context_avg_tps: Optional[float] = None  # first quartile of context
    long_context_avg_tps: Optional[float] = None  # last quartile of context


class MTPSettingComparison(BaseModel):
    """Comparison of different MTP draft-token settings."""

    setting: int
    count: int
    avg_tps: float
    avg_acceptance_rate: float
    avg_context_length: float
    min_tps: float
    max_tps: float
    tps_std: float
    tps_cv: float
    degradation_rate: Optional[float] = None
    # Context range for comparable-context uplift calculation
    min_context: int = 0
    max_context: int = 0


class AnalysisOutput(BaseModel):
    """Output of the analyze stage."""

    run_id: str = ""
    metrics: AnalysisMetrics = Field(default_factory=AnalysisMetrics)
    mtp_setting_comparisons: list[MTPSettingComparison] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class Recommendation(BaseModel):
    """A single recommendation with reasoning."""

    mtp_setting: int
    avg_throughput_uptick: Optional[float] = None  # percentage vs baseline
    long_context_efficiency: str = ""  # good, moderate, poor
    stability: str = ""  # stable, moderate, unstable
    memory_overhead_estimate_mb: Optional[float] = None
    reasoning: list[str] = Field(default_factory=list)


class RecommendationOutput(BaseModel):
    """Output of the recommend stage."""

    run_id: str = ""
    recommended_setting: int = 0
    recommended: Recommendation = Field(default_factory=Recommendation)
    all_recommendations: list[Recommendation] = Field(default_factory=list)
    summary_text: str = ""


class PlotConfig(BaseModel):
    """Configuration for chart generation."""

    title: str = "MTP Profiler - Throughput & Acceptance Rate"
    smoothing_window: int = 5
    show_trendline: bool = True
    show_baseline: bool = True
    use_lowess: bool = False
    frac_lowess: float = 0.33
    output_format: str = "png"  # png, svg, pdf
    dpi: int = 150
    figure_width: int = 12
    figure_height: int = 7
