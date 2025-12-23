"""Tests for R² score calculation functionality.

Test-Driven Development: Tests written before implementation.
Tests cover R² score computation for multi-output PINN validation.
"""

import numpy as np
import pytest

from pinn.validation.r2_score import R2ScoreCalculator


class TestR2ScoreCalculator:
    """Test R² score calculation for PINN validation."""

    def test_r2_score_perfect_prediction(self):
        """Test R² = 1.0 for perfect predictions."""
        calculator = R2ScoreCalculator()

        # Perfect predictions
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        r2 = calculator.compute_r2(y_true, y_pred)

        assert r2 == pytest.approx(1.0, abs=1e-6)

    def test_r2_score_mean_prediction(self):
        """Test R² = 0.0 when prediction equals mean."""
        calculator = R2ScoreCalculator()

        # Predictions are always the mean
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full(5, np.mean(y_true))

        r2 = calculator.compute_r2(y_true, y_pred)

        assert r2 == pytest.approx(0.0, abs=1e-6)

    def test_r2_score_worse_than_mean(self):
        """Test R² < 0.0 when prediction is worse than mean."""
        calculator = R2ScoreCalculator()

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Bad predictions (opposite sign)
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        r2 = calculator.compute_r2(y_true, y_pred)

        assert r2 < 0.0

    def test_r2_score_good_prediction(self):
        """Test R² between 0 and 1 for reasonable predictions."""
        calculator = R2ScoreCalculator()

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Good but not perfect predictions
        y_pred = np.array([1.1, 2.1, 2.9, 3.9, 5.1])

        r2 = calculator.compute_r2(y_true, y_pred)

        assert 0.0 < r2 < 1.0
        assert r2 > 0.9  # Should be high quality

    def test_r2_score_multi_output(self):
        """Test R² calculation for multiple outputs (T1, T3, Ux, Uy)."""
        calculator = R2ScoreCalculator()

        N = 100
        # Create 4 output fields
        y_true = {
            'T1': np.random.randn(N),
            'T3': np.random.randn(N),
            'Ux': np.random.randn(N),
            'Uy': np.random.randn(N)
        }

        # Create predictions with varying quality
        y_pred = {
            'T1': y_true['T1'] + 0.01 * np.random.randn(N),  # High quality
            'T3': y_true['T3'] + 0.01 * np.random.randn(N),
            'Ux': y_true['Ux'] + 0.01 * np.random.randn(N),
            'Uy': y_true['Uy'] + 0.01 * np.random.randn(N)
        }

        r2_scores = calculator.compute_r2_multi_output(y_true, y_pred)

        # Should return dict with 4 entries
        assert isinstance(r2_scores, dict)
        assert set(r2_scores.keys()) == {'T1', 'T3', 'Ux', 'Uy'}

        # All should be high quality
        for field, r2 in r2_scores.items():
            assert r2 > 0.9, f"{field} R² should be > 0.9, got {r2}"

    def test_r2_score_multi_output_varying_quality(self):
        """Test R² scores with varying quality across outputs."""
        calculator = R2ScoreCalculator()

        N = 100
        y_true = {
            'T1': np.random.randn(N),
            'T3': np.random.randn(N),
            'Ux': np.random.randn(N),
            'Uy': np.random.randn(N)
        }

        # Create predictions with different quality
        y_pred = {
            'T1': y_true['T1'] + 0.01 * np.random.randn(N),  # Excellent
            'T3': y_true['T3'] + 0.1 * np.random.randn(N),   # Good
            'Ux': y_true['Ux'] + 0.5 * np.random.randn(N),   # Fair
            'Uy': np.full(N, np.mean(y_true['Uy']))          # Poor (mean)
        }

        r2_scores = calculator.compute_r2_multi_output(y_true, y_pred)

        # T1 should be best
        assert r2_scores['T1'] > r2_scores['T3']
        assert r2_scores['T3'] > r2_scores['Ux']
        assert r2_scores['Ux'] > r2_scores['Uy']

        # Uy should be close to 0 (predicting mean)
        assert r2_scores['Uy'] == pytest.approx(0.0, abs=0.1)

    def test_r2_score_uses_sklearn_formula(self):
        """Test that R² = 1 - SS_res / SS_tot formula is used."""
        calculator = R2ScoreCalculator()

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9, 5.1])

        r2 = calculator.compute_r2(y_true, y_pred)

        # Manually compute R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        expected_r2 = 1 - (ss_res / ss_tot)

        assert r2 == pytest.approx(expected_r2, abs=1e-6)

    def test_r2_score_handles_constant_true_values(self):
        """Test R² when true values are constant (SS_tot = 0)."""
        calculator = R2ScoreCalculator()

        # Constant true values
        y_true = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])

        r2 = calculator.compute_r2(y_true, y_pred)

        # Should return 1.0 for perfect prediction of constant
        assert r2 == pytest.approx(1.0, abs=1e-6)

    def test_r2_score_raises_error_for_mismatched_shapes(self):
        """Test that ValueError is raised for mismatched array shapes."""
        calculator = R2ScoreCalculator()

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])  # Different length

        with pytest.raises(ValueError, match="Shape mismatch"):
            calculator.compute_r2(y_true, y_pred)

    def test_r2_score_raises_error_for_empty_arrays(self):
        """Test that ValueError is raised for empty arrays."""
        calculator = R2ScoreCalculator()

        y_true = np.array([])
        y_pred = np.array([])

        with pytest.raises(ValueError, match="Empty arrays"):
            calculator.compute_r2(y_true, y_pred)

    def test_r2_score_multi_output_raises_error_for_missing_fields(self):
        """Test that KeyError is raised if fields don't match."""
        calculator = R2ScoreCalculator()

        y_true = {'T1': np.array([1.0, 2.0, 3.0])}
        y_pred = {'T1': np.array([1.0, 2.0, 3.0]), 'T3': np.array([1.0, 2.0, 3.0])}

        with pytest.raises(KeyError, match="Field mismatch"):
            calculator.compute_r2_multi_output(y_true, y_pred)

    def test_r2_score_format_report(self):
        """Test formatted report generation for R² scores."""
        calculator = R2ScoreCalculator()

        r2_scores = {
            'T1': 0.95,
            'T3': 0.92,
            'Ux': 0.88,
            'Uy': 0.85
        }

        report = calculator.format_report(r2_scores)

        # Should contain field names and scores
        assert 'T1' in report
        assert 'T3' in report
        assert 'Ux' in report
        assert 'Uy' in report
        assert '0.95' in report or '0.950' in report

    def test_r2_score_format_report_with_warnings(self):
        """Test that report includes warnings for low R² scores."""
        calculator = R2ScoreCalculator()

        r2_scores = {
            'T1': 0.95,
            'T3': 0.85,  # Below 0.9 threshold
            'Ux': 0.75,  # Below 0.9 threshold
            'Uy': 0.92
        }

        report = calculator.format_report(r2_scores, threshold=0.9)

        # Should contain warnings
        assert 'WARNING' in report or 'warning' in report.lower()
        assert 'T3' in report  # Low score field
        assert 'Ux' in report  # Low score field
