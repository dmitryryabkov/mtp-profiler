"""Chart generation for MTP Profiler.

Generates publication-quality charts for MTP performance analysis:
- Generation throughput vs context length with trendline
- Draft acceptance rate over context length
- Throughput stability boxplot
- Uplift vs baseline charts
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from mtp_profiler.models.schemas import (
    AnalysisOutput,
    ProfileOutput,
    PlotConfig,
)

logger = logging.getLogger(__name__)

# Color palette for different runs - more differentiated colors
RUN_COLORS = {
    "steelblue": "#4A90E2",
    "coral": "#FF6B6B",
    "green": "#4ECDC4",
    "purple": "#9B59B6",
    "navy": "#2C3E50",
}


def plot(
    profile: ProfileOutput,
    analysis: AnalysisOutput | None = None,
    output_dir: Path | str = ".",
    config: PlotConfig | None = None,
    run_id: str | None = None,
) -> list[Path]:
    """Generate charts from parsed/analyzed data.

    Args:
        profile: Parsed profile output.
        analysis: Optional pre-computed analysis.
        output_dir: Directory to save charts.
        config: Optional chart configuration.
        run_id: Optional run ID to plot.

    Returns:
        List of generated file paths.
    """
    if config is None:
        config = PlotConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_id is None and profile.runs:
        run_id = profile.runs[0].id

    if analysis is None:
        from mtp_profiler.analyzer.analyzer import analyze as do_analyze
        analysis = do_analyze(profile, run_id)

    paths: list[Path] = []

    # Generate main chart: throughput + acceptance rate (merged)
    chart_path = _plot_throughput_and_acceptance(
        profile, run_id, config, output_dir
    )
    if chart_path:
        paths.append(chart_path)

    # Generate uplift vs baseline chart if there are multiple settings
    if len(analysis.mtp_setting_comparisons) > 1:
        uplift_path = _plot_uplift_vs_baseline(
            profile, analysis, config, output_dir
        )
        if uplift_path:
            paths.append(uplift_path)

    # Generate stability chart
    stability_path = _plot_stability_boxplot(
        profile, run_id, config, output_dir
    )
    if stability_path:
        paths.append(stability_path)

    return paths


def _get_df_for_run(profile: ProfileOutput, run_id: str) -> pd.DataFrame:
    """Extract measurements for a specific run as a DataFrame."""
    run = None
    for r in profile.runs:
        if r.id == run_id:
            run = r
            break
    if run is None:
        return pd.DataFrame()

    records = []
    for m in run.measurements:
        record = {
            "n_tokens": m.n_tokens,
            "gen_tps": m.generation_tokens_per_second,
            "prompt_tps": m.prompt_tokens_per_second,
            "acceptance_rate": m.draft_acceptance_rate,
            "eval_ms": m.eval_time_ms,
            "total_ms": m.total_time_ms,
            "n_decoded": m.n_decoded,
            "run_id": run.id,
            "mtp_n_max": run.metadata.mtp_config.get("n_max", None),
        }
        records.append(record)

    return pd.DataFrame(records)


def _get_all_runs_df(profile: ProfileOutput) -> pd.DataFrame:
    """Extract measurements from all runs as a single DataFrame."""
    all_records = []
    for run in profile.runs:
        run_df = _get_df_for_run(profile, run.id)
        all_records.append(run_df)

    if not all_records:
        return pd.DataFrame()

    return pd.concat(all_records, ignore_index=True)


def _smooth_rolling(series: pd.Series, window: int) -> pd.Series:
    """Apply rolling average smoothing."""
    return series.rolling(window=max(window, 1), min_periods=1, center=True).mean()


def _smooth_lowess(series: pd.Series, frac: float) -> pd.Series:
    """Apply LOWESS smoothing.

    Args:
        series: Input series (assumed sorted by index).
        frac: Fraction of data to use for each local regression.

    Returns:
        Smoothed values as numpy array.
    """
    try:
        from statsmodels.nonparametric.smoothing_lowess import lowess
        clean = series.dropna().sort_index()
        if len(clean) < 3:
            return series
        smoothed = lowess(clean.values, clean.index, frac=frac, return_sorted=False)
        result = pd.Series(np.full(len(series), np.nan), index=series.index)
        result[clean.index] = smoothed
        return result
    except ImportError:
        logger.warning("statsmodels not installed, falling back to rolling average")
        return _smooth_rolling(series, 5)
    except Exception:
        logger.warning("LOWESS smoothing failed, falling back to rolling average")
        return _smooth_rolling(series, 5)


def _plot_throughput_and_acceptance(
    profile: ProfileOutput,
    run_id: str,
    config: PlotConfig,
    output_dir: Path,
) -> Optional[Path]:
    """Plot generation throughput and acceptance rate vs context length.

    Groups all runs by MTP n_max setting, merging runs with the same setting
    into a single dataset per setting.
    """
    df = _get_all_runs_df(profile)
    if df.empty:
        logger.warning("No data for throughput/acceptance plot")
        return None

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(config.figure_width, config.figure_height * 1.2)
    )
    fig.suptitle(config.title, fontsize=14, fontweight="bold", y=1.02)

    # Sort by context length
    df = df.sort_values("n_tokens")

    # Color palette for different settings
    color_values = list(RUN_COLORS.values())

    # Group by mtp_n_max instead of run_id
    # Baseline runs (no MTP) get n_max=None, MTP runs get their n_max value
    df_for_grouping = df.copy()
    df_for_grouping["_group_key"] = df_for_grouping["mtp_n_max"].apply(
        lambda x: f"n_max_{int(x)}" if pd.notna(x) else "baseline"
    )

    # --- Top subplot: Throughput vs Context ---
    has_any_data = False
    group_list = list(df_for_grouping.groupby("_group_key"))
    for i, (group_key, group_df) in enumerate(group_list):
        valid_tp = group_df[group_df["gen_tps"].notna() & group_df["n_tokens"].notna()].copy()
        if valid_tp.empty:
            continue
        has_any_data = True
        x = valid_tp["n_tokens"].values.astype(float)
        y = valid_tp["gen_tps"].values.astype(float)
        color = color_values[i % len(color_values)]

        # Determine label
        if group_key == "baseline":
            label = "Baseline (no MTP)"
        else:
            n_max = int(group_key.split("_")[2])
            label = f"n_max={n_max}"

        # Scatter plot
        ax1.scatter(x, y, alpha=0.4, s=20, color=color, zorder=3, label=label)

        # Rolling average smooth
        if len(x) >= 3:
            sorted_idx = np.argsort(x)
            if config.use_lowess:
                smooth_y = _smooth_lowess(
                    pd.Series(y, index=sorted_idx), config.frac_lowess
                ).values
            else:
                smooth_y = _smooth_rolling(
                    pd.Series(y)[sorted_idx], config.smoothing_window
                ).values
            ax1.plot(x[sorted_idx], smooth_y, color=color,
                     linewidth=2, alpha=0.8, zorder=4)

        # Linear trendline
        if config.show_trendline and len(x) >= 3:
            try:
                coeffs = np.polyfit(x, y, 1)
                trendline = np.poly1d(coeffs)
                slope_per_1000 = coeffs[0] * 1000
                ax1.plot(x, trendline(x), color=color, linestyle="--",
                         linewidth=1.5, alpha=0.7, zorder=5,
                         label=f"{label} trend: {slope_per_1000:+.2f} t/s per 1k tokens")
            except (np.linalg.LinAlgError, ValueError):
                pass

    ax1.set_xlabel("Context Length (tokens)", fontsize=11)
    ax1.set_ylabel("Generation Throughput (tokens/sec)", fontsize=11)
    ax1.grid(True, alpha=0.3, zorder=1)
    ax1.legend(fontsize=9, loc="best")

    # --- Bottom subplot: Acceptance Rate vs Context ---
    has_any_acc = False
    for i, (group_key, group_df) in enumerate(group_list):
        if group_key == "baseline":
            continue

        valid_acc = group_df[group_df["acceptance_rate"].notna() & group_df["n_tokens"].notna()].copy()
        if valid_acc.empty:
            continue

        has_any_acc = True
        x = valid_acc["n_tokens"].values.astype(float)
        y = valid_acc["acceptance_rate"].values.astype(float) * 100
        color = color_values[i % len(color_values)]
        label = f"n_max={int(group_key.split('_')[2])}"

        # Scatter plot
        ax2.scatter(x, y, alpha=0.4, s=20, color=color, zorder=3, label=label)

        # Rolling average smooth
        if len(x) >= 3:
            sorted_idx = np.argsort(x)
            if config.use_lowess:
                smooth_y = _smooth_lowess(
                    pd.Series(y, index=sorted_idx), config.frac_lowess
                ).values
            else:
                smooth_y = _smooth_rolling(
                    pd.Series(y)[sorted_idx], config.smoothing_window
                ).values
            ax2.plot(x[sorted_idx], smooth_y, color=color,
                     linewidth=2, alpha=0.8, zorder=4)

        # Mean line per group
        if config.show_baseline:
            mean_acc = valid_acc["acceptance_rate"].mean() * 100
            ax2.axhline(y=mean_acc, color=color, linestyle=":",
                        linewidth=1, alpha=0.8,
                        label=f"{label} mean: {mean_acc:.1f}%")

    if has_any_acc:
        ax2.set_xlabel("Context Length (tokens)", fontsize=11)
        ax2.set_ylabel("Draft Acceptance Rate (%)", fontsize=11)
        ax2.set_title("Draft Acceptance Rate vs Context Length", fontsize=12)
        ax2.grid(True, alpha=0.3, zorder=1)
        ax2.legend(fontsize=9, loc="best")
    else:
        ax2.set_xlabel("Context Length (tokens)", fontsize=11)
        ax2.set_ylabel("Draft Acceptance Rate (%)", fontsize=11)
        ax2.set_title("Draft Acceptance Rate vs Context Length", fontsize=12)
        ax2.text(0.5, 0.5, "No MTP runs with acceptance data",
                ha="center", va="center", transform=ax2.transAxes,
                fontsize=12, alpha=0.5)
        ax2.grid(True, alpha=0.3, zorder=1)

    fig.tight_layout()
    path = output_dir / f"throughput_and_acceptance.{config.output_format}"
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart: %s", path)
    return path


def _plot_uplift_vs_baseline(
    profile: ProfileOutput,
    analysis: AnalysisOutput,
    config: PlotConfig,
    output_dir: Path,
) -> Optional[Path]:
    """Plot throughput uplift vs baseline across MTP settings."""
    if len(analysis.mtp_setting_comparisons) < 2:
        return None

    fig, ax = plt.subplots(
        figsize=(config.figure_width, config.figure_height)
    )

    comparisons = analysis.mtp_setting_comparisons
    labels = [f"Draft Tokens {c.setting}" for c in comparisons]
    avg_tps = [c.avg_tps for c in comparisons]
    std_tps = [c.tps_std for c in comparisons]

    baseline = avg_tps[0]
    uplift_pct = [(t - baseline) / baseline * 100 for t in avg_tps]

    x = np.arange(len(comparisons))
    bars = ax.bar(
        x,
        uplift_pct,
        yerr=std_tps,
        capsize=5,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    for i, bar in enumerate(bars):
        bar.set_color("green" if uplift_pct[i] > 0 else "red")

    ax.axhline(y=0, color="black", linewidth=0.8, zorder=1)
    ax.set_xlabel("MTP Setting", fontsize=12)
    ax.set_ylabel("Throughput Uplift vs Baseline (%)", fontsize=12)
    ax.set_title("MTP Setting Comparison: Throughput Uplift",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3, axis="y", zorder=1)

    for i, (bar, val) in enumerate(zip(bars, uplift_pct)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + (1 if val >= 0 else -1),
            f"{val:+.1f}%",
            ha="center", va="bottom" if val >= 0 else "top",
            fontsize=10,
        )

    fig.tight_layout()
    path = output_dir / f"uplift_vs_baseline.{config.output_format}"
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart: %s", path)
    return path


def _plot_stability_boxplot(
    profile: ProfileOutput,
    run_id: str,
    config: PlotConfig,
    output_dir: Path,
) -> Optional[Path]:
    """Plot throughput distribution as boxplot grouped by MTP setting."""
    df = _get_all_runs_df(profile)
    if df.empty or "gen_tps" not in df.columns:
        logger.warning("No data for stability boxplot")
        return None

    df = df[df["gen_tps"].notna()].copy()
    if df.empty:
        return None

    # Group by mtp_n_max instead of run_id
    df["_group_key"] = df["mtp_n_max"].apply(
        lambda x: f"n_max_{int(x)}" if pd.notna(x) else "baseline"
    )

    fig, ax = plt.subplots(
        figsize=(config.figure_width, config.figure_height)
    )

    group_groups = []
    labels = []
    color_values = list(RUN_COLORS.values())

    for i, (group_key, group) in enumerate(df.groupby("_group_key")):
        tps_values = group["gen_tps"].dropna().values
        if len(tps_values) > 0:
            group_groups.append(tps_values)
            if group_key == "baseline":
                labels.append("Baseline (no MTP)")
            else:
                n_max = int(group_key.split("_")[2])
                labels.append(f"n_max={n_max}")

    if group_groups:
        bp = ax.boxplot(
            group_groups,
            labels=labels,
            patch_artist=True,
            showmeans=True,
            meanline=True,
        )

        for j, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(color_values[j % len(color_values)])
            patch.set_alpha(0.7)

    ax.set_xlabel("MTP Setting", fontsize=12)
    ax.set_ylabel("Generation Throughput (tokens/sec)", fontsize=12)
    ax.set_title("Throughput Stability by MTP Setting", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", zorder=1)

    fig.tight_layout()
    path = output_dir / f"stability_boxplot.{config.output_format}"
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart: %s", path)
    return path
