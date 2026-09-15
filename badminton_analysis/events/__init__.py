"""Post-processing for badminton events and user-facing reports."""

from .rally_analyzer import analyze_match, generate_match_report

__all__ = ["analyze_match", "generate_match_report"]

