"""
Proactive Monitoring Agent

Tracks query history and confidence scores in memory.
Proactively surfaces flagged low-confidence queries and generates
a monitoring report identifying potential knowledge gaps in the corpus.

In production this would persist to a database and run on a schedule.
For demonstration, it maintains an in-memory log and exposes a /monitor endpoint.
"""

from datetime import datetime


class MonitorAgent:
    """
    Monitors query patterns and confidence scores across sessions.
    Identifies knowledge gaps and surfaces actionable intelligence
    without being asked.
    """

    def __init__(self, low_confidence_threshold: float = 0.6):
        self.threshold = low_confidence_threshold
        self.query_log: list[dict] = []
        self.flagged_queries: list[dict] = []

    def record(
        self,
        question: str,
        confidence_score: float,
        grounding_status: str,
        flagged: bool,
    ) -> None:
        """
        Records a completed query cycle into the monitor log.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "confidence_score": round(confidence_score, 3),
            "grounding_status": grounding_status,
            "flagged": flagged,
        }
        self.query_log.append(entry)

        if flagged:
            self.flagged_queries.append(entry)

    def get_report(self) -> dict:
        """
        Generates a proactive monitoring report surfacing:
        - Total queries processed
        - Flagged (low-confidence) query rate
        - Average confidence score
        - Recent flagged queries for review
        - Actionable recommendation
        """
        total = len(self.query_log)
        flagged_count = len(self.flagged_queries)
        flag_rate = round(flagged_count / total, 3) if total > 0 else 0.0

        avg_confidence = (
            round(sum(q["confidence_score"] for q in self.query_log) / total, 3)
            if total > 0
            else 0.0
        )

        grounding_distribution: dict[str, int] = {}
        for q in self.query_log:
            status = q["grounding_status"]
            grounding_distribution[status] = grounding_distribution.get(status, 0) + 1

        return {
            "summary": {
                "total_queries": total,
                "flagged_queries": flagged_count,
                "flag_rate": flag_rate,
                "average_confidence_score": avg_confidence,
            },
            "grounding_distribution": grounding_distribution,
            "recent_flagged_queries": self.flagged_queries[-5:],
            "recommendation": self._generate_recommendation(flag_rate, avg_confidence),
        }

    def _generate_recommendation(self, flag_rate: float, avg_confidence: float) -> str:
        """
        Generates a plain-English recommendation based on current metrics.
        """
        if flag_rate == 0.0:
            return "No issues detected. Knowledge corpus is performing well."
        if flag_rate >= 0.4:
            return (
                f"High flag rate of {flag_rate:.0%} with average confidence {avg_confidence:.2f}. "
                "Recommend reviewing recent flagged queries and expanding the knowledge corpus "
                "to cover missing topics."
            )
        return (
            f"Moderate flag rate of {flag_rate:.0%}. "
            "Review recent flagged queries to identify knowledge gaps."
        )
