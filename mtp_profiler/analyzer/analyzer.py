"""Analysis engine for MTP Profiler.

Computes derived metrics from parsed telemetry data:
- Throughput statistics (avg, std, min, max, median)
- Context-length correlation and degradation rate
- MTP setting comparisons
- Stability metrics (variance, coefficient of variation)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from mtp_profiler.models.schemas import (
    AnalysisMetrics,
    AnalysisOutput,
    MTPSettingComparison,
    ProfileOutput,
    Run,
)

logger = logging.getLogger(__name__)


def analyze(profile: ProfileOutput, run_id: str | None = None) -> AnalysisOutput:
    """Analyze parsed profiling data and compute derived metrics.

    Args:
        profile: Parsed profile output from the parse stage.
        run_id: Optional specific run ID to analyze. If None and multiple
                runs exist, merges all runs and groups by MTP n_max setting.

    Returns:
        AnalysisOutput with computed metrics.
    """
    if not profile.runs:
        return AnalysisOutput(
            metrics=AnalysisMetrics(),
            summary={"error": "No runs to analyze"},
        )

    # If explicit run_id requested, analyze that single run
    if run_id:
        target_run = None
        for r in profile.runs:
            if r.id == run_id:
                target_run = r
                break
        if target_run is None:
            return AnalysisOutput(
                metrics=AnalysisMetrics(),
                summary={"error": f"Run not found: {run_id}"},
            )
        return _analyze_run(target_run)

    # If a single run, analyze it directly
    if len(profile.runs) == 1:
        return _analyze_run(profile.runs[0])

    # Multiple runs: merge all measurements and group by MTP n_max
    return _analyze_all(profile)


def _analyze_run(run: Run) -> AnalysisOutput:
    """Analyze a single run's measurements."""
    df = _measurements_to_df(run)

    if df.empty:
        return AnalysisOutput(
            run_id=run.id,
            metrics=AnalysisMetrics(),
            summary={"warning": "No measurements to analyze"},
        )

    metrics = _compute_metrics(df)
    comparisons = _compute_mtp_comparisons(df, run.metadata.mtp_config)

    # Build summary
    summary = _build_summary(metrics, comparisons, df)

    return AnalysisOutput(
        run_id=run.id,
        metrics=metrics,
        mtp_setting_comparisons=comparisons,
        summary=summary,
    )


def _collect_all_measurements(profile: ProfileOutput) -> list[dict]:
    """Collect all measurements from all runs, tagging each with n_max."""
    records = []
    for run in profile.runs:
        n_max = run.metadata.mtp_config.get("n_max", None)
        for m in run.measurements:
            record = {
                "n_tokens": m.n_tokens,
                "n_decoded": m.n_decoded,
                "prompt_tps": m.prompt_tokens_per_second,
                "gen_tps": m.generation_tokens_per_second,
                "acceptance_rate": m.draft_acceptance_rate,
                "prompt_eval_ms": m.prompt_eval_time_ms,
                "eval_ms": m.eval_time_ms,
                "total_ms": m.total_time_ms,
                "n_drafts_gen": m.n_drafts_generated,
                "n_drafts_acc": m.n_drafts_accepted,
                "n_max": n_max,
            }
            records.append(record)
    return records


def _analyze_all(profile: ProfileOutput) -> AnalysisOutput:
    """Analyze all runs merged, grouped by MTP n_max setting."""
    records = _collect_all_measurements(profile)
    if not records:
        return AnalysisOutput(
            metrics=AnalysisMetrics(),
            summary={"warning": "No measurements across any runs to analyze"},
        )

    df = pd.DataFrame(records)

    # Compute aggregate metrics across all runs
    metrics = _compute_metrics(df)

    # Group by n_max for MTP comparisons
    comparisons = _compute_cross_run_mtp_comparisons(df)

    summary = _build_summary(metrics, comparisons, df)

    return AnalysisOutput(
        metrics=metrics,
        mtp_setting_comparisons=comparisons,
        summary=summary,
    )


def _compute_cross_run_mtp_comparisons(df: pd.DataFrame) -> list[MTPSettingComparison]:
    """Group measurements by n_max across all runs and compute comparisons."""
    if "n_max" not in df.columns:
        return []

    # Separate baseline (no MTP) and MTP runs
    baseline_df = df[df["n_max"].isna()]
    mtp_df = df[df["n_max"].notna()]

    comparisons = []

    # Baseline group (no MTP)
    if not baseline_df.empty:
        comp = _single_group_comparison(baseline_df, setting=0)
        comparisons.insert(0, comp)

    # MTP groups by n_max
    if not mtp_df.empty:
        grouped = mtp_df.groupby("n_max")
        for n_max, group in grouped:
            comp = _single_group_comparison(group, setting=int(n_max))
            comparisons.append(comp)

    return comparisons


def _measurements_to_df(run: Run) -> pd.DataFrame:
    """Convert measurements to a pandas DataFrame for analysis."""
    records = []
    for m in run.measurements:
        record = {
            "n_tokens": m.n_tokens,
            "n_decoded": m.n_decoded,
            "prompt_tps": m.prompt_tokens_per_second,
            "gen_tps": m.generation_tokens_per_second,
            "acceptance_rate": m.draft_acceptance_rate,
            "prompt_eval_ms": m.prompt_eval_time_ms,
            "eval_ms": m.eval_time_ms,
            "total_ms": m.total_time_ms,
            "n_drafts_gen": m.n_drafts_generated,
            "n_drafts_acc": m.n_drafts_accepted,
        }
        records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df


def _compute_metrics(df: pd.DataFrame) -> AnalysisMetrics:
    """Compute aggregate metrics from a measurement DataFrame."""
    metrics = AnalysisMetrics()

    if df.empty:
        return metrics

    # Generation throughput stats
    gen_tps = df["gen_tps"].dropna()
    if len(gen_tps) > 0:
        metrics.avg_generation_tps = float(gen_tps.mean())
        metrics.std_generation_tps = float(gen_tps.std()) if len(gen_tps) > 1 else 0.0
        metrics.min_generation_tps = float(gen_tps.min())
        metrics.max_generation_tps = float(gen_tps.max())
        metrics.median_generation_tps = float(gen_tps.median())
        metrics.tps_variance = float(gen_tps.var())
        if len(gen_tps) > 1:
            metrics.tps_cv = float(gen_tps.std() / gen_tps.mean())
            metrics.p10_generation_tps = float(gen_tps.quantile(0.10))
            metrics.p90_generation_tps = float(gen_tps.quantile(0.90))

    # Prompt throughput stats
    prompt_tps = df["prompt_tps"].dropna()
    if len(prompt_tps) > 0:
        metrics.avg_prompt_tps = float(prompt_tps.mean())
        metrics.std_prompt_tps = float(prompt_tps.std()) if len(prompt_tps) > 1 else 0.0

    # Acceptance rate stats
    acc = df["acceptance_rate"].dropna()
    if len(acc) > 0:
        metrics.avg_acceptance_rate = float(acc.mean())
        metrics.std_acceptance_rate = float(acc.std()) if len(acc) > 1 else 0.0

    # Context-length correlation with throughput
    valid_ctx = df[df["n_tokens"].notna() & df["gen_tps"].notna()]
    if len(valid_ctx) >= 3:
        n_tokens_arr = valid_ctx["n_tokens"].values.astype(float)
        tps_arr = valid_ctx["gen_tps"].values.astype(float)
        corr = np.corrcoef(n_tokens_arr, tps_arr)[0, 1]
        metrics.context_tps_correlation = float(corr) if not np.isnan(corr) else None

        # Linear regression for degradation rate (tps per 1000 tokens)
        try:
            from scipy import stats as scipy_stats

            slope, _, _, _, _ = scipy_stats.linregress(n_tokens_arr, tps_arr)
            metrics.context_degradation_rate = float(slope * 1000)  # per 1000 tokens
        except ImportError:
            # scipy not available, fallback to two-point estimate
            sorted_idx = np.argsort(n_tokens_arr)
            if len(sorted_idx) >= 2:
                first = tps_arr[sorted_idx[0]]
                last = tps_arr[sorted_idx[-1]]
                ctx_diff = (n_tokens_arr[sorted_idx[-1]] - n_tokens_arr[sorted_idx[0]]) / 1000
                if ctx_diff > 0:
                    metrics.context_degradation_rate = float((last - first) / ctx_diff)
        except ValueError:
            # Fallback: simple two-point estimate
            sorted_idx = np.argsort(n_tokens_arr)
            if len(sorted_idx) >= 2:
                first = tps_arr[sorted_idx[0]]
                last = tps_arr[sorted_idx[-1]]
                ctx_diff = (n_tokens_arr[sorted_idx[-1]] - n_tokens_arr[sorted_idx[0]]) / 1000
                if ctx_diff > 0:
                    metrics.context_degradation_rate = float((last - first) / ctx_diff)

    # MTP draft stats
    gen_drafts = df["n_drafts_gen"].dropna()
    acc_drafts = df["n_drafts_acc"].dropna()
    if len(gen_drafts) > 0:
        metrics.avg_drafts_per_generation = float(gen_drafts.mean())
    if len(acc_drafts) > 0:
        metrics.avg_accepted_per_generation = float(acc_drafts.mean())

    # Long-context vs short-context avg TPS
    ctx = df["n_tokens"].dropna()
    if len(ctx) >= 4:
        q1_threshold = ctx.quantile(0.25)
        q3_threshold = ctx.quantile(0.75)
        short_df = df[(df["n_tokens"].notna()) & (df["n_tokens"] <= q1_threshold)]
        long_df = df[(df["n_tokens"].notna()) & (df["n_tokens"] >= q3_threshold)]
        short_tps = short_df["gen_tps"].dropna()
        long_tps = long_df["gen_tps"].dropna()
        if len(short_tps) > 0:
            metrics.short_context_avg_tps = float(short_tps.mean())
        if len(long_tps) > 0:
            metrics.long_context_avg_tps = float(long_tps.mean())

    return metrics


def _compute_mtp_comparisons(
    df: pd.DataFrame,
    mtp_config: dict | None = None,
) -> list[MTPSettingComparison]:
    """Group measurements by MTP configuration and compare.

    Groups by n_max_draft (the MTP draft-token count) when available.
    Falls back to a single group if all measurements share the same config.
    """
    comparisons = []

    # Try to group by the MTP n_max setting if available in measurements
    if "n_max_draft" in df.columns and df["n_max_draft"].notna().any():
        grouped = df.groupby("n_max_draft")
        for n_max, group in grouped:
            comp = _single_group_comparison(group, setting=int(n_max))
            comparisons.append(comp)
        if comparisons:
            return comparisons

    # If we have metadata with MTP config, use n_max from there
    if mtp_config and "n_max" in mtp_config:
        n_max = int(mtp_config["n_max"])
        comp = _single_group_comparison(df, setting=n_max)
        comparisons.append(comp)
        return comparisons

    # Fallback: single group for all measurements
    if not df.empty:
        comp = _single_group_comparison(df, setting=0)
        comparisons.append(comp)

    return comparisons


def _single_group_comparison(group: pd.DataFrame, setting: int) -> MTPSettingComparison:
    """Compute comparison metrics for a single group of measurements."""
    gen_tps = group["gen_tps"].dropna()
    acc = group["acceptance_rate"].dropna()
    ctx = group["n_tokens"].dropna()

    # Collect raw (context_length, tps) points where both are valid
    valid = group[["n_tokens", "gen_tps"]].dropna()
    raw_points = [(float(row["n_tokens"]), float(row["gen_tps"])) for _, row in valid.iterrows()]

    return MTPSettingComparison(
        setting=setting,
        count=len(gen_tps),
        avg_tps=float(gen_tps.mean()) if len(gen_tps) > 0 else 0.0,
        avg_acceptance_rate=float(acc.mean()) if len(acc) > 0 else 0.0,
        avg_context_length=float(ctx.mean()) if len(ctx) > 0 else 0.0,
        min_tps=float(gen_tps.min()) if len(gen_tps) > 0 else 0.0,
        max_tps=float(gen_tps.max()) if len(gen_tps) > 0 else 0.0,
        tps_std=float(gen_tps.std()) if len(gen_tps) > 1 else 0.0,
        tps_cv=float(gen_tps.std() / gen_tps.mean()) if len(gen_tps) > 1 and gen_tps.mean() > 0 else 0.0,
        min_context=int(ctx.min()) if len(ctx) > 0 else 0,
        max_context=int(ctx.max()) if len(ctx) > 0 else 0,
        raw_points=raw_points,
    )


def _build_summary(
    metrics: AnalysisMetrics,
    comparisons: list[MTPSettingComparison],
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Build a human-readable summary dict."""
    summary: dict[str, Any] = {}

    if metrics.avg_generation_tps:
        summary["avg_generation_tps"] = round(metrics.avg_generation_tps, 2)
    if metrics.avg_acceptance_rate:
        summary["avg_acceptance_rate"] = round(metrics.avg_acceptance_rate * 100, 2)
    if metrics.context_tps_correlation is not None:
        summary["context_tps_correlation"] = round(metrics.context_tps_correlation, 4)
    if metrics.context_degradation_rate is not None:
        summary["context_degradation_rate_per_1k"] = round(
            metrics.context_degradation_rate, 4
        )

    # Stability assessment
    if metrics.tps_cv:
        if metrics.tps_cv < 0.05:
            summary["stability"] = "stable"
        elif metrics.tps_cv < 0.15:
            summary["stability"] = "moderate"
        else:
            summary["stability"] = "unstable"

    # Long-context behavior
    if metrics.short_context_avg_tps and metrics.long_context_avg_tps:
        ratio = metrics.long_context_avg_tps / metrics.short_context_avg_tps
        if ratio > 0.9:
            summary["long_context_behavior"] = "good"
        elif ratio > 0.7:
            summary["long_context_behavior"] = "moderate"
        else:
            summary["long_context_behavior"] = "degraded"

    # Number of measurements
    summary["total_measurements"] = len(df)

    return summary
