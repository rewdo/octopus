"""
Octopus — Multi-Brain Agent Infrastructure.

A token-economic, never-forgetting, self-evolving cognitive operating system
for AI agents. Like an octopus with multiple brains, different tasks are
handled by specialized cognitive modules coordinated by a central router.

Core principles:
    - Token is scarce currency — cloud LLM calls are the last resort
    - Local-first execution — rules → skills → small models → RAG → compressed cloud
    - Skill > Prompt — reusable skills are the primary asset
    - Memory ≠ Context — memory is stored; context is compiled on demand
    - Verify everything — no single LLM output is trusted without validation
"""

__version__ = "0.1.0"
__author__ = "Octopus Contributors"
__license__ = "MIT"

from octopus.config import OctopusConfig
from octopus.router import CognitiveRouter
from octopus.agent import OctopusAgent

__all__ = ["OctopusConfig", "CognitiveRouter", "OctopusAgent"]
