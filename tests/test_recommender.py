"""Tests for the recommendation engine."""

import pytest

from mtp_profiler.models.schemas import (
    AnalysisMetrics,
    AnalysisOutput,
    MTPSettingComparison,
    Recommendation,
    RecommendationOutput,
)
from mtp_profiler.recommender.recommender import (
    recommend,
    _score_setting,
    _select_best,
    _assess_long_context_efficiency,
    _assess_stability,
)


class TestRecommend:
    """Tests for the recommend function."""

    def test_recommend_single_setting(self):
        """Test recommendation with a single MTP setting."""
        analysis = AnalysisOutput(
            run_id="test",
            metrics=AnalysisMetrics(
                avg_generation_tps=30.0,
                avg_acceptance_rate=0.9,
            ),
            mtp_setting_comparisons=[
                MTPSettingComparison(
                    setting=2,
                    count=10,
                    avg_tps=30.0,
                    avg_acceptance_rate=0.9,
                    avg_context_length=5000.0,
                    min_tps=25.0,
                    max_tps=35.0,
                    tps_std=3.0,
                    tps_cv=0.1,
                ),
            ],
        )

        result = recommend(None, analysis)

        assert result.recommended_setting == 2
        assert result.summary_text != ""

    def test_recommend_multiple_settings(self):
        """Test recommendation with multiple MTP settings."""
        analysis = AnalysisOutput(
            run_id="test",
            metrics=AnalysisMetrics(
                avg_generation_tps=28.0,
                avg_acceptance_rate=0.85,
            ),
            mtp_setting_comparisons=[
                MTPSettingComparison(
                    setting=1,
                    count=5,
                    avg_tps=25.0,
                    avg_acceptance_rate=0.95,
                    avg_context_length=5000.0,
                    min_tps=20.0,
                    max_tps=30.0,
                    tps_std=3.0,
                    tps_cv=0.12,
                ),
                MTPSettingComparison(
                    setting=2,
                    count=5,
                    avg_tps=30.0,
                    avg_acceptance_rate=0.85,
                    avg_context_length=5000.0,
                    min_tps=25.0,
                    max_tps=35.0,
                    tps_std=3.0,
                    tps_cv=0.1,
                ),
            ],
        )

        result = recommend(None, analysis)

        # Setting 2 should win due to higher throughput
        assert result.recommended_setting == 2

    def test_recommend_empty_comparisons(self):
        """Test recommendation with no data."""
        analysis = AnalysisOutput(run_id="test")
        result = recommend(None, analysis)

        assert result.summary_text == "Not enough data to make a recommendation."

    def test_recommend_with_profile(self):
        """Test recommendation with both profile and analysis."""
        from mtp_profiler.models.schemas import ProfileOutput, Run, RunMetadata, Measurement

        profile = ProfileOutput(
            runs=[
                Run(
                    id="test",
                    metadata=RunMetadata(
                        model="test.gguf",
                        mtp_config={"n_max": 2},
                    ),
                    measurements=[
                        Measurement(generation_tokens_per_second=30.0),
                    ],
                ),
            ],
        )

        analysis = AnalysisOutput(
            run_id="test",
            metrics=AnalysisMetrics(avg_generation_tps=30.0),
            mtp_setting_comparisons=[
                MTPSettingComparison(
                    setting=2,
                    count=1,
                    avg_tps=30.0,
                    avg_acceptance_rate=0.9,
                    avg_context_length=1000.0,
                    min_tps=30.0,
                    max_tps=30.0,
                    tps_std=0.0,
                    tps_cv=0.0,
                ),
            ],
        )

        result = recommend(profile, analysis)
        assert result.run_id == "test"


class TestScoring:
    """Tests for the scoring algorithm."""

    def test_high_throughput_scores_higher(self):
        """Test that higher throughput gets a better score."""
        rec = Recommendation(mtp_setting=1)
        comp_high = MTPSettingComparison(
            setting=1, count=10, avg_tps=35.0, avg_acceptance_rate=0.9,
            avg_context_length=5000.0, min_tps=30.0, max_tps=40.0,
            tps_std=2.0, tps_cv=0.05,
        )
        comp_low = MTPSettingComparison(
            setting=1, count=10, avg_tps=25.0, avg_acceptance_rate=0.9,
            avg_context_length=5000.0, min_tps=20.0, max_tps=30.0,
            tps_std=2.0, tps_cv=0.05,
        )

        all_comps = [comp_high, comp_low]
        score_high = _score_setting(rec, comp_high, all_comps)
        score_low = _score_setting(rec, comp_low, all_comps)

        assert score_high > score_low

    def test_stable_setting_scores_higher(self):
        """Test that more stable settings get better scores."""
        rec = Recommendation(mtp_setting=1)
        comp_stable = MTPSettingComparison(
            setting=1, count=10, avg_tps=30.0, avg_acceptance_rate=0.9,
            avg_context_length=5000.0, min_tps=28.0, max_tps=32.0,
            tps_std=1.0, tps_cv=0.03,
        )
        comp_unstable = MTPSettingComparison(
            setting=1, count=10, avg_tps=30.0, avg_acceptance_rate=0.9,
            avg_context_length=5000.0, min_tps=15.0, max_tps=45.0,
            tps_std=5.0, tps_cv=0.17,
        )

        score_stable = _score_setting(rec, comp_stable)
        score_unstable = _score_setting(rec, comp_unstable)

        assert score_stable > score_unstable

    def test_high_acceptance_scores_higher(self):
        """Test that higher acceptance rates get better scores."""
        rec = Recommendation(mtp_setting=1)
        comp_high_acc = MTPSettingComparison(
            setting=1, count=10, avg_tps=30.0, avg_acceptance_rate=0.95,
            avg_context_length=5000.0, min_tps=25.0, max_tps=35.0,
            tps_std=2.0, tps_cv=0.05,
        )
        comp_low_acc = MTPSettingComparison(
            setting=1, count=10, avg_tps=30.0, avg_acceptance_rate=0.70,
            avg_context_length=5000.0, min_tps=25.0, max_tps=35.0,
            tps_std=2.0, tps_cv=0.05,
        )

        score_high = _score_setting(rec, comp_high_acc)
        score_low = _score_setting(rec, comp_low_acc)

        assert score_high > score_low


class TestSelectBest:
    """Tests for the best setting selection."""

    def test_selects_highest_scored(self):
        """Test that the highest-scoring setting is selected."""
        recs = [
            Recommendation(mtp_setting=1, long_context_efficiency="good", stability="stable"),
            Recommendation(mtp_setting=2, long_context_efficiency="poor", stability="unstable"),
        ]
        comps = [
            MTPSettingComparison(
                setting=1, count=10, avg_tps=30.0, avg_acceptance_rate=0.9,
                avg_context_length=5000.0, min_tps=28.0, max_tps=32.0,
                tps_std=1.0, tps_cv=0.03,
            ),
            MTPSettingComparison(
                setting=2, count=10, avg_tps=32.0, avg_acceptance_rate=0.7,
                avg_context_length=5000.0, min_tps=15.0, max_tps=45.0,
                tps_std=5.0, tps_cv=0.16,
            ),
        ]

        best = _select_best(recs, comps)
        assert best.mtp_setting == 1  # Setting 1 should win due to stability and efficiency


class TestAssessments:
    """Tests for individual assessment functions."""

    def test_long_context_good(self):
        """Test good long-context efficiency assessment."""
        from mtp_profiler.models.schemas import AnalysisMetrics

        metrics = AnalysisMetrics(
            short_context_avg_tps=35.0,
            long_context_avg_tps=32.0,
        )
        result = _assess_long_context_efficiency(
            MTPSettingComparison(setting=0, count=10, avg_tps=30.0, avg_acceptance_rate=0.9,
                                 avg_context_length=5000.0, min_tps=25.0, max_tps=35.0,
                                 tps_std=3.0, tps_cv=0.1),
            metrics,
        )
        assert result == "good"  # 32/35 = 0.91 > 0.9

    def test_long_context_poor(self):
        """Test poor long-context efficiency assessment."""
        from mtp_profiler.models.schemas import AnalysisMetrics

        metrics = AnalysisMetrics(
            short_context_avg_tps=35.0,
            long_context_avg_tps=24.0,
        )
        result = _assess_long_context_efficiency(
            MTPSettingComparison(setting=0, count=10, avg_tps=30.0, avg_acceptance_rate=0.9,
                                 avg_context_length=5000.0, min_tps=25.0, max_tps=35.0,
                                 tps_std=3.0, tps_cv=0.1),
            metrics,
        )
        assert result == "degraded"  # 24/35 = 0.69 < 0.7

    def test_stability_stable(self):
        """Test stable throughput assessment."""
        comp = MTPSettingComparison(
            setting=0, count=10, avg_tps=30.0, avg_acceptance_rate=0.9,
            avg_context_length=5000.0, min_tps=28.0, max_tps=32.0,
            tps_std=1.0, tps_cv=0.03,
        )
        assert _assess_stability(comp) == "stable"

    def test_stability_variable(self):
        """Test variable throughput assessment."""
        comp = MTPSettingComparison(
            setting=0, count=10, avg_tps=30.0, avg_acceptance_rate=0.9,
            avg_context_length=5000.0, min_tps=15.0, max_tps=45.0,
            tps_std=8.0, tps_cv=0.27,
        )
        assert _assess_stability(comp) == "variable"

    def test_stability_unknown_cv(self):
        """Test stability with zero CV."""
        comp = MTPSettingComparison(
            setting=0, count=10, avg_tps=30.0, avg_acceptance_rate=0.9,
            avg_context_length=5000.0, min_tps=30.0, max_tps=30.0,
            tps_std=0.0, tps_cv=0.0,
        )
        assert _assess_stability(comp) == "stable"
