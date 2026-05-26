"""
Octopus Memory System — four-layer cognitive memory architecture.

Layers:
  L1: Working Memory   (transient, current session)
  L2: Episodic Memory  (what happened, timeline)
  L3: Semantic Memory  (facts, preferences, knowledge)
  L4: Procedural Memory (skills, workflows)

Plus the Context Compiler that assembles relevant context for each brain request.
"""

from __future__ import annotations

from octopus.memory.memory_graph import MemoryGraph, MemoryNode
from octopus.memory.layers import (
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
)
from octopus.memory.context_compiler import ContextCompiler, TaskAnalysis

__all__ = [
    # Graph
    "MemoryGraph",
    "MemoryNode",
    # Layers
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    # Compiler
    "ContextCompiler",
    "TaskAnalysis",
]
