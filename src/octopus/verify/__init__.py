"""
Verification Layer for Octopus multi-brain architecture.

The Verification Layer is the quality gate between any brain's output
and the user. Every BrainResponse passes through verification before
it reaches the CLI, API, or any other output channel.

Components:
    Verifier       — Unified entry point, orchestrates sub-verifiers
    CodeVerifier   — Python code syntax & safety analysis
    FactVerifier   — Hallucination & contradiction detection (Phase 1)
"""

from .verifier import Verifier, CodeVerifier, FactVerifier, VerificationResult

__all__ = ["Verifier", "CodeVerifier", "FactVerifier", "VerificationResult"]
