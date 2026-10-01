"""
Attack Risk Estimator and Severity Classifier.
Translates continuous model probabilities into graded risk levels with operational confidence.
"""

from typing import Dict, Any, List


class RiskEstimator:
    """Manages attack probability interpretation and alert severity classification."""

    def __init__(
        self,
        low_threshold: float = 0.30,
        medium_threshold: float = 0.60,
        high_threshold: float = 0.80
    ):
        self.low = low_threshold
        self.medium = medium_threshold
        self.high = high_threshold

    def classify_severity(self, probability: float) -> str:
        """Categorizes continuous probability into risk tier."""
        p = float(probability)
        if p < self.low:
            return "LOW"
        elif p < self.medium:
            return "MEDIUM"
        elif p < self.high:
            return "HIGH"
        else:
            return "CRITICAL"

    def format_risk_timeline(
        self,
        current_probability: float,
        future_probabilities: List[float],
        window_duration: float = 10.0
    ) -> List[Dict[str, Any]]:
        """
        Creates an annotated timeline of current and projected risk.
        """
        timeline = []
        # Current state step 0
        timeline.append({
            "step": 0,
            "time_offset_seconds": 0.0,
            "label": "Current State S(t)",
            "attack_probability": round(float(current_probability), 4),
            "severity": self.classify_severity(current_probability)
        })

        for i, prob in enumerate(future_probabilities, start=1):
            timeline.append({
                "step": i,
                "time_offset_seconds": i * window_duration,
                "label": f"+{i} window ({i * int(window_duration)}s)",
                "attack_probability": round(float(prob), 4),
                "severity": self.classify_severity(prob)
            })

        return timeline
