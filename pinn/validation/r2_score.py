"""R² score calculation for PINN model validation.

This module provides R² (coefficient of determination) calculation for evaluating
PINN prediction quality against FDTD ground truth data.
"""

import numpy as np


class R2ScoreCalculator:
    """Calculate R² scores for PINN validation.

    R² score measures the proportion of variance in the dependent variable
    that is predictable from the independent variable.

    R² = 1 - SS_res / SS_tot

    where:
        SS_res = Σ(y_true - y_pred)²  (residual sum of squares)
        SS_tot = Σ(y_true - mean(y_true))²  (total sum of squares)

    R² = 1.0: Perfect prediction
    R² = 0.0: Prediction equals mean (baseline)
    R² < 0.0: Prediction worse than mean
    """

    def compute_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² score for single output.

        Args:
            y_true: Ground truth values (N,)
            y_pred: Predicted values (N,)

        Returns:
            R² score (float)

        Raises:
            ValueError: If arrays are empty or have mismatched shapes

        Preconditions:
            - y_true and y_pred must have same shape
            - Arrays must not be empty

        Postconditions:
            - Returns R² score
            - R² = 1.0 for perfect prediction
            - R² = 0.0 when prediction equals mean

        Example:
            >>> calculator = R2ScoreCalculator()
            >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> y_pred = np.array([1.1, 2.1, 2.9, 3.9, 5.1])
            >>> r2 = calculator.compute_r2(y_true, y_pred)
            >>> r2 > 0.9  # High quality prediction
            True
        """
        # Validate inputs
        if y_true.size == 0 or y_pred.size == 0:
            raise ValueError("Empty arrays not allowed")

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            )

        # Compute residual sum of squares
        ss_res = np.sum((y_true - y_pred) ** 2)

        # Compute total sum of squares
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        # Handle constant true values (SS_tot = 0)
        if ss_tot == 0:
            # If predictions are also constant and equal to true values, R² = 1
            if ss_res == 0:
                return 1.0
            # If predictions differ from constant true values, R² = -inf
            # But we return 0.0 for numerical stability
            return 0.0

        # Compute R² score
        r2 = 1.0 - (ss_res / ss_tot)

        return r2

    def compute_r2_multi_output(
        self,
        y_true: dict[str, np.ndarray],
        y_pred: dict[str, np.ndarray]
    ) -> dict[str, float]:
        """Compute R² scores for multiple outputs (T1, T3, Ux, Uy).

        Args:
            y_true: Dict of ground truth arrays {field_name: array}
            y_pred: Dict of predicted arrays {field_name: array}

        Returns:
            Dict of R² scores {field_name: r2_score}

        Raises:
            KeyError: If field names don't match between y_true and y_pred

        Preconditions:
            - y_true and y_pred must have same keys
            - Each array pair must have same shape

        Postconditions:
            - Returns R² score for each field
            - All fields present in both dicts are evaluated

        Example:
            >>> calculator = R2ScoreCalculator()
            >>> y_true = {
            ...     'T1': np.array([1.0, 2.0, 3.0]),
            ...     'T3': np.array([4.0, 5.0, 6.0])
            ... }
            >>> y_pred = {
            ...     'T1': np.array([1.1, 2.0, 2.9]),
            ...     'T3': np.array([4.1, 5.0, 5.9])
            ... }
            >>> r2_scores = calculator.compute_r2_multi_output(y_true, y_pred)
            >>> all(r2 > 0.9 for r2 in r2_scores.values())
            True
        """
        # Validate field names match
        if set(y_true.keys()) != set(y_pred.keys()):
            raise KeyError(
                f"Field mismatch: y_true has {set(y_true.keys())}, "
                f"y_pred has {set(y_pred.keys())}"
            )

        # Compute R² for each field
        r2_scores = {}
        for field_name in y_true.keys():
            r2 = self.compute_r2(y_true[field_name], y_pred[field_name])
            r2_scores[field_name] = r2

        return r2_scores

    def format_report(
        self,
        r2_scores: dict[str, float],
        threshold: float = 0.9
    ) -> str:
        """Format R² scores as a human-readable report.

        Args:
            r2_scores: Dict of R² scores {field_name: r2_score}
            threshold: Warning threshold for low R² scores (default: 0.9)

        Returns:
            Formatted report string

        Example:
            >>> calculator = R2ScoreCalculator()
            >>> r2_scores = {'T1': 0.95, 'T3': 0.85, 'Ux': 0.92, 'Uy': 0.88}
            >>> report = calculator.format_report(r2_scores)
            >>> 'T1' in report and '0.95' in report
            True
        """
        lines = []
        lines.append("=" * 50)
        lines.append("R² Score Validation Report")
        lines.append("=" * 50)
        lines.append("")

        # Sort by field name for consistent output
        for field_name in sorted(r2_scores.keys()):
            r2 = r2_scores[field_name]
            status = "✓" if r2 >= threshold else "⚠"
            lines.append(f"{status} {field_name:4s}: R² = {r2:.4f}")

        lines.append("")

        # Add warnings for low scores
        low_scores = {k: v for k, v in r2_scores.items() if v < threshold}
        if low_scores:
            lines.append("WARNING: Low R² scores detected!")
            lines.append(f"Fields below threshold ({threshold}):")
            for field_name, r2 in sorted(low_scores.items()):
                lines.append(f"  - {field_name}: {r2:.4f}")
            lines.append("")
            lines.append("Recommendation: Consider adjusting hyperparameters")
            lines.append("  - Increase network depth/width")
            lines.append("  - Increase training iterations")
            lines.append("  - Adjust learning rate")
            lines.append("  - Check PDE loss weighting")
        else:
            lines.append(f"✓ All fields meet quality threshold (R² >= {threshold})")

        lines.append("")
        lines.append("=" * 50)

        return "\n".join(lines)
