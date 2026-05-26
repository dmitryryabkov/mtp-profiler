"""Tests for the analysis engine."""

import json
import tempfile
from pathlib import Path

import pytest

from mtp_profiler.models.schemas import ProfileOutput
from mtp_profiler.parser.log_parser import parse_log
from mtp_profiler.analyzer.analyzer import analyze, _compute_metrics, _measurements_to_df
from tests.fixtures import PARSED_JSON_FIXTURE


class TestAnalyze:
    """Tests for the analyze function."""

    def test_analyze_sample_data(self):
        """Test analysis on the parsed sample log."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("""\x1b[34m10.54.639.911\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 | prompt eval time =    8119.49 ms /  3138 tokens (    2.59 ms per token,   386.48 tokens per second)
\x1b[34m10.54.639.914\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 |        eval time =   41904.01 ms /  1291 tokens (   32.46 ms per token,    30.81 tokens per second)
\x1b[34m10.54.639.916\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 | draft acceptance = 0.89809 (  705 accepted /   785 generated)
\x1b[34m11.32.600.886\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 2 | prompt eval time =   29831.76 ms / 11199 tokens (    2.66 ms per token,   375.41 tokens per second)
\x1b[34m11.32.600.889\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 2 |        eval time =    8043.07 ms /   264 tokens (   30.47 ms per token,    32.82 tokens per second)
\x1b[34m11.32.600.890\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 2 | draft acceptance = 0.94767 (  163 accepted /   172 generated)
""")
            f.flush()
            profile = parse_log(f.name)

        analysis = analyze(profile)

        assert analysis.run_id == "run_1"
        assert analysis.metrics.avg_generation_tps is not None
        assert analysis.metrics.avg_acceptance_rate is not None
        assert len(analysis.mtp_setting_comparisons) >= 1
        assert "avg_generation_tps" in analysis.summary

    def test_analyze_empty_profile(self):
        """Test analysis on an empty profile."""
        profile = ProfileOutput()
        analysis = analyze(profile)

        assert "error" in analysis.summary

    def test_analyze_with_run_id(self):
        """Test analysis with explicit run_id."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("""\x1b[34m10.54.639.914\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 |        eval time =   41904.01 ms /  1291 tokens (   32.46 ms per token,    30.81 tokens per second)
""")
            f.flush()
            profile = parse_log(f.name)

        analysis = analyze(profile, run_id="run_1")
        assert analysis.run_id == "run_1"

    def test_analyze_invalid_run_id(self):
        """Test analysis with non-existent run_id."""
        profile = ProfileOutput()
        analysis = analyze(profile, run_id="nonexistent")
        assert "error" in analysis.summary

    def test_mtp_grouping_by_config(self):
        """Test that measurements are grouped by MTP config, not task_id."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("""\x1b[34m0.00.052.030\x1b[0m \x1b[32mI \x1b[0m  - MTL0    : Apple M3 Pro (28753 MiB, 28753 MiB free)
\x1b[34m0.02.499.412\x1b[0m \x1b[32mI \x1b[0mcommon_speculative_impl_draft_mtp: - n_max=2, n_min=0, p_min=0.70
\x1b[34m10.54.639.911\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 | prompt eval time =    8119.49 ms /  3138 tokens (    2.59 ms per token,   386.48 tokens per second)
\x1b[34m10.54.639.914\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 |        eval time =   41904.01 ms /  1291 tokens (   32.46 ms per token,    30.81 tokens per second)
\x1b[34m10.54.639.916\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 | draft acceptance = 0.89809 (  705 accepted /   785 generated)
\x1b[34m11.32.600.889\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 999 |        eval time =    8043.07 ms /   264 tokens (   30.47 ms per token,    32.82 tokens per second)
\x1b[34m11.32.600.890\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 999 | draft acceptance = 0.94767 (  163 accepted /   172 generated)
""")
            f.flush()
            profile = parse_log(f.name)

        analysis = analyze(profile)

        # Should have exactly 1 MTP setting group (n_max=2), not 2 task groups
        assert len(analysis.mtp_setting_comparisons) == 1
        comparison = analysis.mtp_setting_comparisons[0]
        assert comparison.setting == 2
        assert comparison.count == 2  # Both measurements in one group

    def test_json_roundtrip(self):
        """Test that analysis output can be serialized and deserialized."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("""\x1b[34m10.54.639.914\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 |        eval time =   41904.01 ms /  1291 tokens (   32.46 ms per token,    30.81 tokens per second)
""")
            f.flush()
            profile = parse_log(f.name)

        analysis = analyze(profile)
        json_str = json.dumps(analysis.model_dump(mode="json"), default=str)
        loaded = type(analysis).model_validate(json.loads(json_str))
        assert loaded.run_id == analysis.run_id
        assert loaded.metrics.avg_generation_tps == analysis.metrics.avg_generation_tps


class TestComputeMetrics:
    """Tests for the _compute_metrics function."""

    def test_basic_throughput_stats(self):
        """Test basic throughput statistics computation."""
        from mtp_profiler.models.schemas import Measurement, Run, RunMetadata
        from mtp_profiler.analyzer.analyzer import _measurements_to_df, _compute_metrics

        run = Run(
            id="test",
            measurements=[
                Measurement(generation_tokens_per_second=30.0),
                Measurement(generation_tokens_per_second=32.0),
                Measurement(generation_tokens_per_second=28.0),
            ],
        )

        df = _measurements_to_df(run)
        metrics = _compute_metrics(df)

        assert metrics.avg_generation_tps == pytest.approx(30.0)
        assert metrics.min_generation_tps == 28.0
        assert metrics.max_generation_tps == 32.0
        assert metrics.median_generation_tps == pytest.approx(30.0)

    def test_acceptance_rate_stats(self):
        """Test acceptance rate statistics."""
        from mtp_profiler.models.schemas import Measurement, Run
        from mtp_profiler.analyzer.analyzer import _measurements_to_df, _compute_metrics

        run = Run(
            id="test",
            measurements=[
                Measurement(draft_acceptance_rate=0.9),
                Measurement(draft_acceptance_rate=0.95),
                Measurement(draft_acceptance_rate=0.85),
            ],
        )

        df = _measurements_to_df(run)
        metrics = _compute_metrics(df)

        assert metrics.avg_acceptance_rate == pytest.approx(0.9)

    def test_context_correlation(self):
        """Test context-length correlation computation."""
        from mtp_profiler.models.schemas import Measurement, Run
        from mtp_profiler.analyzer.analyzer import _measurements_to_df, _compute_metrics

        run = Run(
            id="test",
            measurements=[
                Measurement(n_tokens=1000, generation_tokens_per_second=35.0),
                Measurement(n_tokens=5000, generation_tokens_per_second=30.0),
                Measurement(n_tokens=10000, generation_tokens_per_second=25.0),
                Measurement(n_tokens=20000, generation_tokens_per_second=20.0),
            ],
        )

        df = _measurements_to_df(run)
        metrics = _compute_metrics(df)

        # Should have negative correlation (more context = slower)
        assert metrics.context_tps_correlation is not None
        assert metrics.context_tps_correlation < 0

    def test_empty_dataframe(self):
        """Test metrics computation on empty DataFrame."""
        from mtp_profiler.models.schemas import Run
        from mtp_profiler.analyzer.analyzer import _measurements_to_df, _compute_metrics

        run = Run(id="test", measurements=[])
        df = _measurements_to_df(run)
        metrics = _compute_metrics(df)

        # All stats should be None for empty data
        assert metrics.avg_generation_tps is None


class TestMeasurementsToDf:
    """Tests for DataFrame conversion."""

    def test_conversion(self):
        """Test that measurements are correctly converted to DataFrame."""
        from mtp_profiler.models.schemas import Measurement, Run
        from mtp_profiler.analyzer.analyzer import _measurements_to_df

        run = Run(
            id="test",
            measurements=[
                Measurement(
                    n_tokens=1000,
                    generation_tokens_per_second=30.0,
                    draft_acceptance_rate=0.9,
                ),
            ],
        )

        df = _measurements_to_df(run)
        assert len(df) == 1
        assert df["n_tokens"].iloc[0] == 1000
        assert df["gen_tps"].iloc[0] == 30.0
        assert df["acceptance_rate"].iloc[0] == 0.9

    def test_null_handling(self):
        """Test that null fields are handled correctly."""
        from mtp_profiler.models.schemas import Measurement, Run
        from mtp_profiler.analyzer.analyzer import _measurements_to_df

        run = Run(
            id="test",
            measurements=[
                Measurement(generation_tokens_per_second=30.0),
                Measurement(),  # All nulls
            ],
        )

        df = _measurements_to_df(run)
        assert len(df) == 2
        assert df["gen_tps"].iloc[0] == 30.0
        assert df["gen_tps"].iloc[1] is None or df["gen_tps"].iloc[1] != df["gen_tps"].iloc[0]


class TestCrossRunAnalysis:
    """Tests for cross-run analysis with merged measurements."""

    def test_cross_run_merges_same_n_max(self):
        """Test that multiple runs with same n_max are merged into one group."""
        from mtp_profiler.models.schemas import ProfileOutput, Run, RunMetadata, Measurement

        profile = ProfileOutput(
            runs=[
                Run(
                    id="run_1",
                    metadata=RunMetadata(mtp_config={"n_max": 2}),
                    measurements=[
                        Measurement(n_tokens=1000, generation_tokens_per_second=30.0, draft_acceptance_rate=0.9),
                        Measurement(n_tokens=5000, generation_tokens_per_second=28.0, draft_acceptance_rate=0.85),
                    ],
                ),
                Run(
                    id="run_2",
                    metadata=RunMetadata(mtp_config={"n_max": 2}),
                    measurements=[
                        Measurement(n_tokens=2000, generation_tokens_per_second=32.0, draft_acceptance_rate=0.92),
                        Measurement(n_tokens=6000, generation_tokens_per_second=29.0, draft_acceptance_rate=0.88),
                    ],
                ),
            ],
        )

        analysis = analyze(profile)

        # Should have 1 MTP setting group (n_max=2) with 4 measurements total
        assert len(analysis.mtp_setting_comparisons) == 1
        comp = analysis.mtp_setting_comparisons[0]
        assert comp.setting == 2
        assert comp.count == 4

    def test_cross_run_separates_different_n_max(self):
        """Test that runs with different n_max values are separate groups."""
        from mtp_profiler.models.schemas import ProfileOutput, Run, RunMetadata, Measurement

        profile = ProfileOutput(
            runs=[
                Run(
                    id="run_1",
                    metadata=RunMetadata(mtp_config={"n_max": 1}),
                    measurements=[
                        Measurement(n_tokens=1000, generation_tokens_per_second=35.0, draft_acceptance_rate=0.95),
                    ],
                ),
                Run(
                    id="run_2",
                    metadata=RunMetadata(mtp_config={"n_max": 2}),
                    measurements=[
                        Measurement(n_tokens=2000, generation_tokens_per_second=30.0, draft_acceptance_rate=0.85),
                    ],
                ),
            ],
        )

        analysis = analyze(profile)

        # Should have 2 MTP setting groups
        assert len(analysis.mtp_setting_comparisons) == 2
        settings = {c.setting for c in analysis.mtp_setting_comparisons}
        assert settings == {1, 2}

    def test_cross_run_includes_baseline(self):
        """Test that baseline runs (no MTP) are included as group 0."""
        from mtp_profiler.models.schemas import ProfileOutput, Run, RunMetadata, Measurement

        profile = ProfileOutput(
            runs=[
                Run(
                    id="run_1",
                    metadata=RunMetadata(mtp_config={}),
                    measurements=[
                        Measurement(n_tokens=1000, generation_tokens_per_second=35.0),
                    ],
                ),
                Run(
                    id="run_2",
                    metadata=RunMetadata(mtp_config={"n_max": 2}),
                    measurements=[
                        Measurement(n_tokens=2000, generation_tokens_per_second=30.0, draft_acceptance_rate=0.85),
                    ],
                ),
            ],
        )

        analysis = analyze(profile)

        # Should have baseline (0) and n_max=2
        assert len(analysis.mtp_setting_comparisons) == 2
        assert analysis.mtp_setting_comparisons[0].setting == 0
        assert analysis.mtp_setting_comparisons[1].setting == 2

    def test_cross_run_single_run_unchanged(self):
        """Test that single-run analysis is unchanged."""
        from mtp_profiler.models.schemas import ProfileOutput, Run, RunMetadata, Measurement

        profile = ProfileOutput(
            runs=[
                Run(
                    id="run_1",
                    metadata=RunMetadata(mtp_config={"n_max": 2}),
                    measurements=[
                        Measurement(n_tokens=1000, generation_tokens_per_second=30.0, draft_acceptance_rate=0.9),
                    ],
                ),
            ],
        )

        analysis = analyze(profile)

        assert len(analysis.mtp_setting_comparisons) == 1
        assert analysis.mtp_setting_comparisons[0].setting == 2
        assert analysis.mtp_setting_comparisons[0].count == 1

    def test_cross_run_empty_measurements(self):
        """Test cross-run analysis with no measurements."""
        from mtp_profiler.models.schemas import ProfileOutput, Run, RunMetadata

        profile = ProfileOutput(
            runs=[
                Run(id="run_1", metadata=RunMetadata(mtp_config={"n_max": 2})),
                Run(id="run_2", metadata=RunMetadata(mtp_config={"n_max": 2})),
            ],
        )

        analysis = analyze(profile)

        assert "warning" in analysis.summary

    def test_cross_run_collect_all_measurements(self):
        """Test that _collect_all_measurements gathers from all runs."""
        from mtp_profiler.models.schemas import ProfileOutput, Run, RunMetadata, Measurement
        from mtp_profiler.analyzer.analyzer import _collect_all_measurements

        profile = ProfileOutput(
            runs=[
                Run(
                    id="run_1",
                    metadata=RunMetadata(mtp_config={"n_max": 2}),
                    measurements=[
                        Measurement(n_tokens=1000, generation_tokens_per_second=30.0),
                    ],
                ),
                Run(
                    id="run_2",
                    metadata=RunMetadata(mtp_config={"n_max": 2}),
                    measurements=[
                        Measurement(n_tokens=2000, generation_tokens_per_second=32.0),
                        Measurement(n_tokens=3000, generation_tokens_per_second=28.0),
                    ],
                ),
            ],
        )

        records = _collect_all_measurements(profile)
        assert len(records) == 3
        assert all(r["n_max"] == 2 for r in records)
