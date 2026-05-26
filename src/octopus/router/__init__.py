"""
Octopus Cognitive Router — the central decision engine.

The router receives user requests, scores them across 9 dimensions,
and routes to the most appropriate brain based on weighted thresholds.
"""

from .cognitive_router import CognitiveRouter, RouterDecision

__all__ = ["CognitiveRouter", "RouterDecision"]
