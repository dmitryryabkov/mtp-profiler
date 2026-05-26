"""Deterministic recommendation engine for MTP Profiler.

Evaluates parsed and analyzed telemetry data to recommend
optimal MTP (Multi-Token Prediction) draft-token settings.

The engine is:
- Deterministic: same input → same output
- Explainable: each recommendation includes reasoning
- Testable: pure functions, no external dependencies
- Simple: clear algorithms, no opaque heuristics
"""

from __future__ import annotations

import logging
from typing import Any

from mtp_profiler.models.schemas import (
    AnalysisMetrics,
    AnalysisOutput,
    MTPSettingComparison,
    ProfileOutput,
    Recommendation,
    RecommendationOutput,
)

logger = logging.getLogger(__name__)


def _compute_comparable_context_uplift(
    baseline_comp: MTPSettingComparison,
    candidate_comp: MTPSettingComparison,
) -> float | None:
    """Compute throughput uplift within the overlapping context range.

    Filters both baseline and candidate raw points to only those within
    the overlapping context range, then computes average TPS for each
    and returns the percentage uplift.

    Returns:
        Uplift percentage, or None if no overlap or insufficient data.
    """
    if not baseline_comp.raw_points or not candidate_comp.raw_points:
        return None

    overlap_min = max(baseline_comp.min_context, candidate_comp.min_context)
    overlap_max = min(baseline_comp.max_context, candidate_comp.max_context)

    if overlap_min >= overlap_max:
        return None

    # Filter points to overlapping range
    baseline_overlap = [tps for ctx, tps in baseline_comp.raw_points
                        if overlap_min <= ctx <= overlap_max]
    candidate_overlap = [tps for ctx, tps in candidate_comp.raw_points
                         if overlap_min <= ctx <= overlap_max]

    if len(baseline_overlap) < 2 or len(candidate_overlap) < 2:
        return None

    baseline_avg = sum(baseline_overlap) / len(baseline_overlap)
    candidate_avg = sum(candidate_overlap) / len(candidate_overlap)

    if baseline_avg == 0:
        return None

    return ((candidate_avg - baseline_avg) / baseline_avg) * 100


def recommend(
    profile: ProfileOutput,
    analysis: AnalysisOutput | None = None,
    run_id: str | None = None,
) -> RecommendationOutput:
    """Generate MTP setting recommendations.

    When multiple runs exist and no run_id is specified, merges all runs
    and groups by MTP n_max setting for cross-run comparison.

    Args:
        profile: Parsed profile output from the parse stage.
        analysis: Optional pre-computed analysis. If None, will be computed.
        run_id: Optional run ID to analyze.

    Returns:
        RecommendationOutput with the best setting and reasoning.
    """
    if analysis is None:
        from mtp_profiler.analyzer.analyzer import analyze as do_analyze

        analysis = do_analyze(profile, run_id)

    if run_id is None and profile is not None and profile.runs:
        run_id = profile.runs[0].id

    return _recommend_from_analysis(analysis, run_id or "")


def _recommend_from_analysis(
    analysis: AnalysisOutput, run_id: str
) -> RecommendationOutput:
    """Generate recommendations from pre-computed analysis."""
    comparisons = analysis.mtp_setting_comparisons
    metrics = analysis.metrics

    if not comparisons:
        return RecommendationOutput(
                run_id=run_id,
                recommended_setting=0,
                recommended=Recommendation(mtp_setting=0),
                summary_text="Not enough data to make a recommendation.",
            )

    # Identify baseline: setting 0 (no MTP) first, otherwise lowest n_max
    baseline_comp = None
    for comp in comparisons:
        if comp.setting == 0:
            baseline_comp = comp
            break
    if baseline_comp is None and comparisons:
        # Use lowest n_max as baseline
        sorted_comps = sorted(comparisons, key=lambda c: c.setting)
        baseline_comp = sorted_comps[0]

    baseline_tps = baseline_comp.avg_tps if baseline_comp else None
    baseline_context_range = (
        (baseline_comp.min_context, baseline_comp.max_context)
        if baseline_comp
        else (0, 0)
    )

    # Compute comparable-context uplift for each setting
    comparable_upticks: dict[int, float | None] = {}
    for comp in comparisons:
        if comp.setting != 0 and baseline_comp:
            comparable_upticks[comp.setting] = _compute_comparable_context_uplift(
                baseline_comp, comp
            )
        else:
            comparable_upticks[comp.setting] = None

    # Evaluate each setting
    all_recommendations: list[Recommendation] = []

    for comp in comparisons:
        rec = _evaluate_setting(comp, metrics, baseline_tps, comparisons,
                                baseline_context_range, comparable_upticks.get(comp.setting))
        all_recommendations.append(rec)

    # Select the best setting
    best = _select_best(all_recommendations, comparisons)

    # Build summary text
    summary_text = _build_summary_text(best, all_recommendations, metrics)

    return RecommendationOutput(
        run_id=run_id,
        recommended_setting=best.mtp_setting,
        recommended=best,
        all_recommendations=all_recommendations,
        summary_text=summary_text,
    )


def _evaluate_setting(
    comp: MTPSettingComparison,
    metrics: AnalysisMetrics,
    baseline_tps: float | None,
    all_comparisons: list[MTPSettingComparison],
    baseline_context_range: tuple[int, int] = (0, 0),
    comparable_uptick: float | None = None,
) -> Recommendation:
    """Evaluate a single MTP setting and produce a recommendation."""
    reasoning: list[str] = []

    # Comparable-context throughput uplift vs baseline
    throughput_uptick = None
    is_comparable = True
    if baseline_tps and baseline_tps > 0:
        # Overall uplift (all contexts)
        overall_uplift = (comp.avg_tps - baseline_tps) / baseline_tps * 100

        # Use comparable-context uplift when available
        if comparable_uptick is not None:
            throughput_uptick = round(comparable_uptick, 1)
            reasoning.append(f"Comparable-context uplift: {comparable_uptick:+.1f}%")
            display_uplift = comparable_uptick
        else:
            # No overlapping context range with baseline — cannot fairly compare
            is_comparable = False
            throughput_uptick = round(overall_uplift, 1)
            reasoning.append(f"No overlapping context range with baseline (range: {comp.min_context}-{comp.max_context})")
            reasoning.append(f"Throughput vs baseline (non-comparable): {overall_uplift:+.1f}%")
            display_uplift = overall_uplift

        if display_uplift > 5:
            reasoning.append(f"+{display_uplift:.1f}% throughput vs baseline")
        elif display_uplift > -5:
            reasoning.append(f"~{display_uplift:.1f}% throughput vs baseline (near parity)")
        else:
            reasoning.append(f"{display_uplift:.1f}% throughput vs baseline (degradation)")

    # Diminishing returns penalty for high draft counts
    if comp.setting > 2:
        penalty_points = (comp.setting - 2) * 1.5
        reasoning.append(f"Diminishing returns penalty: -{penalty_points:.1f} pts (n_max={comp.setting})")

    # Long-context efficiency (per-setting fallback)
    efficiency = _assess_long_context_efficiency(comp, metrics)
    reasoning.append(f"Long-context efficiency: {efficiency}")

    # Stability
    stability = _assess_stability(comp)
    reasoning.append(f"Stability: {stability}")

    # Memory overhead estimate (based on draft count)
    memory_mb = _estimate_memory_overhead(comp)
    if memory_mb is not None:
        reasoning.append(f"Estimated memory overhead: ~{memory_mb:.0f} MB")

    return Recommendation(
        mtp_setting=comp.setting,
        avg_throughput_uptick=throughput_uptick,
        long_context_efficiency=efficiency,
        stability=stability,
        memory_overhead_estimate_mb=memory_mb,
        comparable=is_comparable,
        reasoning=reasoning,
    )


def _assess_long_context_efficiency(
    comp: MTPSettingComparison,
    metrics: AnalysisMetrics,
) -> str:
    """Assess how well a setting performs in long-context scenarios.

    Uses per-setting raw points to compute short vs long context TPS ratio.
    Falls back to global metrics if raw points are unavailable.
    """
    # Try per-setting computation from raw points
    if comp.raw_points and len(comp.raw_points) >= 4:
        # Sort by context length
        sorted_points = sorted(comp.raw_points, key=lambda p: p[0])
        n = len(sorted_points)
        q1_idx = n // 4
        q3_idx = 3 * n // 4

        short_tps = [tps for _, tps in sorted_points[:q1_idx]]
        long_tps = [tps for _, tps in sorted_points[q3_idx:]]

        if short_tps and long_tps:
            short_avg = sum(short_tps) / len(short_tps)
            long_avg = sum(long_tps) / len(long_tps)
            if short_avg > 0:
                ratio = long_avg / short_avg
                if ratio >= 0.9:
                    return "good"
                elif ratio >= 0.7:
                    return "moderate"
                else:
                    return "degraded"

    # Fallback: use global metrics if available
    if metrics.short_context_avg_tps and metrics.long_context_avg_tps:
        ratio = metrics.long_context_avg_tps / metrics.short_context_avg_tps
        if ratio >= 0.9:
            return "good"
        elif ratio >= 0.7:
            return "moderate"
        else:
            return "degraded"

    # Fallback: use min/max ratio within this setting
    if comp.max_tps > 0:
        ratio = comp.min_tps / comp.max_tps
        if ratio >= 0.85:
            return "good"
        elif ratio >= 0.65:
            return "moderate"
        else:
            return "degraded"

    return "unknown"


def _assess_stability(comp: MTPSettingComparison) -> str:
    """Assess the stability of a setting's throughput."""
    if comp.tps_cv is None or comp.tps_cv == 0:
        return "stable"
    if comp.tps_cv < 0.05:
        return "stable"
    elif comp.tps_cv < 0.15:
        return "moderate"
    else:
        return "variable"


def _estimate_memory_overhead(comp: MTPSettingComparison) -> float | None:
    """Estimate memory overhead from draft token count.

    Rough estimate: each draft token adds ~model_size * embedding_dim bytes.
    For a 35B model with 2048 embedding dim, each draft is ~272 MB.
    This is a rough heuristic, not exact.
    """
    # We don't have the exact draft count per setting in the comparison,
    # but we can estimate from the acceptance rate and context
    return None  # Placeholder for future improvement


def _select_best(
    recommendations: list[Recommendation],
    comparisons: list[MTPSettingComparison],
) -> Recommendation:
    """Select the best MTP setting based on a scoring algorithm.

    Scoring considers:
    1. Throughput (higher is better)
    2. Stability (lower CV is better)
    3. Long-context efficiency
    4. Acceptance rate (higher is better)
    """
    if not recommendations:
        return Recommendation(mtp_setting=0)

    # Score each recommendation
    scored = []
    for rec, comp in zip(recommendations, comparisons):
        score = _score_setting(rec, comp, comparisons)
        scored.append((score, rec))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _score_setting(rec: Recommendation, comp: MTPSettingComparison, all_comparisons: list[MTPSettingComparison] | None = None) -> float:
    """Compute a composite score for a single setting.

    Score components (normalized to 0-100):
    - Throughput: 40% weight (based on comparable-context uplift vs baseline)
    - Stability: 25% weight
    - Long-context efficiency: 20% weight
    - Acceptance rate: 15% weight
    - Diminishing returns penalty for high draft counts
    """
    # Throughput score (0-100) based on comparable-context uplift
    # Baseline gets 50 (neutral), positive uplift increases, negative decreases
    uplift = rec.avg_throughput_uptick
    if uplift is not None:
        # Scale: +50% uplift = 100, 0% = 50, -50% = 0
        throughput_score = 50 + (uplift / 50) * 50
        throughput_score = max(0, min(100, throughput_score))
    else:
        # Fallback to raw TPS ratio if no uplift available
        if all_comparisons is None:
            all_comparisons = [comp]
        max_tps = max((c.avg_tps for c in all_comparisons), default=1)
        throughput_score = (comp.avg_tps / max_tps) * 100 if max_tps > 0 else 0

    # Stability score (0-100)
    # Lower CV = better
    cv = comp.tps_cv if comp.tps_cv is not None else 0
    if cv <= 0.05:
        stability_score = 100
    elif cv <= 0.15:
        stability_score = 70
    elif cv <= 0.30:
        stability_score = 40
    else:
        stability_score = max(0, 40 - (cv - 0.3) * 200)

    # Long-context efficiency score
    efficiency_map = {"good": 100, "moderate": 60, "degraded": 20, "unknown": 50}
    efficiency_score = efficiency_map.get(rec.long_context_efficiency, 50)

    # Acceptance rate score (0-100)
    # Baseline (setting 0) has no acceptance rate — give neutral score
    if comp.setting == 0:
        acceptance_score = 50
    else:
        acceptance_score = min(100, comp.avg_acceptance_rate * 100)

    # Weighted composite
    composite = (
        throughput_score * 0.40
        + stability_score * 0.25
        + efficiency_score * 0.20
        + acceptance_score * 0.15
    )

    # Non-comparable penalty: settings without overlapping context range with baseline
    if not rec.comparable:
        composite -= 20

    # Diminishing returns penalty: deduct points for n_max > 2
    if comp.setting > 2:
        diminishing_penalty = (comp.setting - 2) * 1.5
        composite -= diminishing_penalty

    return composite


def _build_summary_text(
    best: Recommendation,
    all_recs: list[Recommendation],
    metrics: AnalysisMetrics,
) -> str:
    """Build a human-readable summary of the recommendation."""
    lines = [f"Recommended MTP setting: {best.mtp_setting}"]

   # Show all settings being compared
    if len(all_recs) > 1:
        lines.append("")
        lines.append("Settings compared:")
        for rec in all_recs:
            marker = " <-- recommended" if rec.mtp_setting == best.mtp_setting else ""
            if rec.avg_throughput_uptick is not None:
                tp_str = f"{rec.avg_throughput_uptick:+.1f}%"
            else:
                tp_str = "baseline"
            lines.append(f"  Setting {rec.mtp_setting}: throughput={tp_str}, long-context={rec.long_context_efficiency}, stability={rec.stability}{marker}")
        lines.append("")

    if best.avg_throughput_uptick is not None:
        sign = "+" if best.avg_throughput_uptick > 0 else ""
        lines.append(f"Throughput vs baseline: {sign}{best.avg_throughput_uptick}%")

    lines.append(f"Long-context efficiency: {best.long_context_efficiency}")
    lines.append(f"Stability: {best.stability}")

    if best.memory_overhead_estimate_mb is not None:
        lines.append(f"Estimated memory overhead: ~{best.memory_overhead_estimate_mb:.0f} MB")

    if metrics.avg_generation_tps:
        lines.append(f"Average generation throughput: {metrics.avg_generation_tps:.2f} t/s")

    if metrics.avg_acceptance_rate is not None:
        lines.append(f"Average draft acceptance rate: {metrics.avg_acceptance_rate * 100:.1f}%")

    return "\n".join(lines)
