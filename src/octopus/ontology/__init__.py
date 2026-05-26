"""
Octopus Ontology System — user world-model construction.

Phase 3: PersonalOntology builds individual user profiles from the
memory graph, tracking preferences, projects, decisions, and skill
affinities.  Future phases will add team/collective ontologies.
"""

from __future__ import annotations

from octopus.ontology.personal_ontology import PersonalOntology, UserProfile

__all__ = [
    "PersonalOntology",
    "UserProfile",
]
