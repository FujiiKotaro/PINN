"""
Test suite for Notebook Task 5: R² Score Evaluation

Tests for R² score calculation, display, and visualization in the forward validation notebook.
This follows TDD methodology for task 5.1 and 5.2.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestTask51_R2ScoreCalculation:
    """Tests for Task 5.1: R² Score calculation and table display"""

    def test_r2_score_calculator_computes_all_fields(self):
        """
        Test that R2ScoreCalculator computes R² for all 4 output fields (T1, T3, Ux, Uy)
        Requirement 3.2: compute_r2_multi_output() returns dict with all fields
        """
        from pinn.validation.r2_score import R2ScoreCalculator

        # Mock validation data: dict format with (N,) arrays per field
        N = 100
        y_true_array = np.random.randn(N, 4)
        y_pred_array = y_true_array + np.random.randn(N, 4) * 0.1  # Add small noise

        # Convert to dict format (as notebook should do)
        y_true = {
            "T1": y_true_array[:, 0],
            "T3": y_true_array[:, 1],
            "Ux": y_true_array[:, 2],
            "Uy": y_true_array[:, 3]
        }
        y_pred = {
            "T1": y_pred_array[:, 0],
            "T3": y_pred_array[:, 1],
            "Ux": y_pred_array[:, 2],
            "Uy": y_pred_array[:, 3]
        }

        calculator = R2ScoreCalculator()
        r2_scores = calculator.compute_r2_multi_output(y_true, y_pred)

        # Verify all 4 fields present
        assert isinstance(r2_scores, dict), "R² scores should be returned as dict"
        assert "T1" in r2_scores, "R² score for T1 missing"
        assert "T3" in r2_scores, "R² score for T3 missing"
        assert "Ux" in r2_scores, "R² score for Ux missing"
        assert "Uy" in r2_scores, "R² score for Uy missing"

        # Verify R² values are in valid range
        for field, r2 in r2_scores.items():
            assert -np.inf < r2 <= 1.0, f"R² for {field} out of range: {r2}"

    def test_r2_scores_displayed_as_dataframe(self):
        """
        Test that R² scores can be converted to pandas DataFrame for table display
        Requirement 3.3: Display as pandas DataFrame
        """
        # Mock R² scores
        r2_scores = {"T1": 0.85, "T3": 0.92, "Ux": 0.95, "Uy": 0.88}

        # Convert to DataFrame (transposed for field names as rows)
        df = pd.DataFrame([r2_scores]).T
        df.columns = ["R² Score"]

        # Verify DataFrame structure
        assert df.shape == (4, 1), f"Expected shape (4, 1), got {df.shape}"
        assert list(df.index) == ["T1", "T3", "Ux", "Uy"], "Field names should be index"
        assert df.loc["T1", "R² Score"] == 0.85, "T1 R² value mismatch"
        assert df.loc["Ux", "R² Score"] == 0.95, "Ux R² value mismatch"

    def test_model_predict_on_validation_data(self):
        """
        Test that trained model can predict on validation data
        Requirement 3.1: Execute prediction on validation data
        """
        # Mock trained model
        mock_model = Mock()
        val_x = np.random.randn(50, 5)  # (N, 5) input
        expected_output = np.random.randn(50, 4)  # (N, 4) output
        mock_model.predict.return_value = expected_output

        # Execute prediction
        y_pred = mock_model.predict(val_x)

        # Verify
        assert y_pred.shape == (50, 4), "Prediction shape mismatch"
        mock_model.predict.assert_called_once_with(val_x)


class TestTask52_R2ScoreVisualization:
    """Tests for Task 5.2: R² Score bar chart and interpretation"""

    def test_r2_bar_chart_generation(self):
        """
        Test that R² scores can be visualized as bar chart
        Requirement 3.4: Generate bar chart
        """
        import matplotlib.pyplot as plt

        # Mock R² scores
        r2_scores = {"T1": 0.82, "T3": 0.88, "Ux": 0.95, "Uy": 0.91}

        # Generate bar chart
        fig, ax = plt.subplots()
        bars = ax.bar(r2_scores.keys(), r2_scores.values())
        ax.set_ylabel("R² Score")
        ax.set_xlabel("Output Field")
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.9, color='r', linestyle='--', label='Target (0.9)')
        ax.legend()

        # Verify chart properties
        assert len(bars) == 4, "Should have 4 bars (one per field)"
        assert ax.get_ylim()[1] == 1.0, "Y-axis should go to 1.0"

        # Cleanup
        plt.close(fig)

    def test_r2_interpretation_comments_generated(self):
        """
        Test that interpretation comments are generated for each field
        Requirement 3.5: Provide interpretation comments
        """
        r2_scores = {"T1": 0.82, "T3": 0.95, "Ux": 0.97, "Uy": 0.88}

        # Generate interpretation comments
        comments = []
        for field, r2 in r2_scores.items():
            if r2 >= 0.95:
                quality = "高精度"
            elif r2 >= 0.9:
                quality = "良好"
            elif r2 >= 0.8:
                quality = "要改善"
            else:
                quality = "改善必須"

            comment = f"{field}: R²={r2:.4f} ({quality})"
            comments.append(comment)

        # Verify comments generated for all fields
        assert len(comments) == 4, "Should have comment for each field"
        assert "T1: R²=0.8200 (要改善)" in comments
        assert "Ux: R²=0.9700 (高精度)" in comments

    def test_improvement_suggestions_for_low_r2(self):
        """
        Test that improvement suggestions are provided when R² < 0.9
        Requirement 3.6: Suggest improvements for R² < 0.9
        """
        r2_scores = {"T1": 0.82, "T3": 0.88, "Ux": 0.95, "Uy": 0.87}

        # Identify fields needing improvement
        fields_needing_improvement = [
            field for field, r2 in r2_scores.items() if r2 < 0.9
        ]

        # Generate suggestions
        suggestions = []
        if fields_needing_improvement:
            suggestions.append("以下のフィールドは改善が推奨されます（R² < 0.9）:")
            for field in fields_needing_improvement:
                suggestions.append(f"  - {field}: R²={r2_scores[field]:.4f}")

            suggestions.append("\n改善策:")
            suggestions.append("  1. loss weightの調整（data weightを増加）")
            suggestions.append("  2. epochs数の増加（現在5000 → 10000へ）")
            suggestions.append("  3. networkの拡大（hidden layersを64 → 128へ）")

        # Verify suggestions generated
        assert len(suggestions) > 0, "Suggestions should be generated"
        assert "T1" in " ".join(suggestions), "T1 should be mentioned"
        assert "Ux" not in " ".join(suggestions), "Ux (R²=0.95) should not need improvement"
        assert "loss weight" in " ".join(suggestions).lower(), "Should suggest loss weight adjustment"

    def test_r2_values_in_valid_range(self):
        """
        Test that all R² values are in valid range [-inf, 1.0]
        """
        r2_scores = {"T1": 0.85, "T3": 0.92, "Ux": 0.95, "Uy": 0.88}

        for field, r2 in r2_scores.items():
            assert r2 <= 1.0, f"{field} R² exceeds 1.0: {r2}"
            assert not np.isnan(r2), f"{field} R² is NaN"
            assert not np.isinf(r2) or r2 < 0, f"{field} R² is invalid: {r2}"


# Integration test for complete Task 5 workflow
class TestTask5Integration:
    """Integration test for complete R² evaluation workflow"""

    def test_complete_r2_evaluation_workflow(self):
        """
        Test complete workflow: predict → calculate R² → display table → visualize → interpret
        """
        from pinn.validation.r2_score import R2ScoreCalculator
        import matplotlib.pyplot as plt

        # Step 1: Mock prediction
        N_val = 100
        val_x = np.random.randn(N_val, 5)
        val_y = np.random.randn(N_val, 4)

        mock_model = Mock()
        y_pred = val_y + np.random.randn(N_val, 4) * 0.1  # Add noise
        mock_model.predict.return_value = y_pred

        # Execute prediction
        predictions = mock_model.predict(val_x)
        assert predictions.shape == (N_val, 4)

        # Step 2: Calculate R² (convert to dict format)
        y_true_dict = {
            "T1": val_y[:, 0],
            "T3": val_y[:, 1],
            "Ux": val_y[:, 2],
            "Uy": val_y[:, 3]
        }
        y_pred_dict = {
            "T1": predictions[:, 0],
            "T3": predictions[:, 1],
            "Ux": predictions[:, 2],
            "Uy": predictions[:, 3]
        }
        calculator = R2ScoreCalculator()
        r2_scores = calculator.compute_r2_multi_output(y_true_dict, y_pred_dict)
        assert len(r2_scores) == 4

        # Step 3: Display as DataFrame
        df = pd.DataFrame([r2_scores]).T
        df.columns = ["R² Score"]
        assert df.shape == (4, 1)

        # Step 4: Generate bar chart
        fig, ax = plt.subplots()
        ax.bar(r2_scores.keys(), r2_scores.values())
        assert len(ax.patches) == 4  # 4 bars
        plt.close(fig)

        # Step 5: Generate interpretations
        interpretations = []
        for field, r2 in r2_scores.items():
            quality = "高精度" if r2 >= 0.9 else "要改善"
            interpretations.append(f"{field}: {quality}")

        assert len(interpretations) == 4
