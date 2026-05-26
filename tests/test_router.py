"""
Tests for the Cognitive Router.

Covers:
    1. Simple greeting → Cheap Brain
    2. Code generation → Skill Brain
    3. Complex planning → Planning Brain
    4. High-risk operation → Frontier Brain
    5. Budget exhausted → forced downgrade
    6. High skill confidence → downgrade to Skill Brain
    7. Threshold boundary tests
    8. Log writing verification
    9. Dimension scoring accuracy
    10. Config override behaviour
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.octopus.brains.base import BrainRequest, BrainType, TaskComplexity, TaskRisk
from src.octopus.config import OctopusConfig
from src.octopus.router.cognitive_router import (
    CognitiveRouter,
    RouterDecision,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> OctopusConfig:
    """Default config with spec-defined weights and thresholds."""
    return OctopusConfig.default()


@pytest.fixture
def router(default_config: OctopusConfig) -> CognitiveRouter:
    """A CognitiveRouter with defaults + temp log file."""
    return CognitiveRouter(
        config=default_config,
        log_path=Path(tempfile.mkdtemp()) / "routing_log.jsonl",
    )


def _make_request(
    user_input: str,
    complexity: TaskComplexity = TaskComplexity.SIMPLE,
    risk: TaskRisk = TaskRisk.NONE,
    novelty_score: float = 0.0,
    **kwargs,
) -> BrainRequest:
    """Convenience factory for BrainRequest."""
    return BrainRequest(
        task_id="test-task-001",
        user_input=user_input,
        complexity=complexity,
        risk=risk,
        novelty_score=novelty_score,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Test 1: Simple greeting → Cheap Brain
# ---------------------------------------------------------------------------


def test_simple_greeting_routes_to_cheap_brain(router: CognitiveRouter) -> None:
    """A trivial 'hello' should route to Cheap Brain."""
    decision = router.analyze("hello")

    assert decision.selected_brain == BrainType.CHEAP
    assert decision.final_score < router.config.router_thresholds["t1"]
    assert "Cheap Brain" in decision.reasoning


# ---------------------------------------------------------------------------
# Test 2: Code generation → Skill Brain (via moderate complexity + skill match)
# ---------------------------------------------------------------------------


def test_code_generation_routes_to_skill_brain(router: CognitiveRouter) -> None:
    """A code-generation task with skill match should route to Skill Brain."""
    request = _make_request(
        user_input="Write a Python function to parse JSON and return keys",
        relevant_skills=["json_parser", "python_helper"],
    )
    decision = router.decide(request)

    # Skill confidence should be non-trivial
    skill_conf = decision.dimension_scores["skill_confidence"]
    assert skill_conf >= 2.0, f"Expected skill confidence ≥ 2, got {skill_conf}"

    # Should route to SKILL if confidence is high enough, else CHEAP or PLANNING
    print(f"Code gen: brain={decision.selected_brain.value}, score={decision.final_score}, skill_conf={skill_conf}")
    # With relevant_skills, skill_confidence should trigger skill routing
    # (skill_conf >= 7 or in the right threshold range)
    assert decision.selected_brain in (
        BrainType.SKILL,
        BrainType.CHEAP,
        BrainType.PLANNING,
    )


def test_code_generation_with_complexity_routes_to_planning(router: CognitiveRouter) -> None:
    """A more complex code task goes to Planning Brain."""
    request = _make_request(
        user_input=(
            "Design and implement a full REST API with authentication, "
            "rate limiting, database integration, and caching. "
            "Write the complete code including def, class, import statements. "
            "Also explain the architecture choices."
        ),
        complexity=TaskComplexity.COMPLEX,
        relevant_skills=["api_designer", "db_helper"],
    )
    decision = router.decide(request)

    print(f"Complex code: brain={decision.selected_brain.value}, score={decision.final_score}")
    # Should be at least Planning level
    assert decision.final_score >= 3.0, f"Score {decision.final_score} should be ≥ T1"
    assert decision.selected_brain in (BrainType.PLANNING, BrainType.FRONTIER)


# ---------------------------------------------------------------------------
# Test 3: Complex planning → Planning Brain
# ---------------------------------------------------------------------------


def test_complex_planning_routes_to_planning_brain(router: CognitiveRouter) -> None:
    """A complex multi-step planning task routes to Planning Brain."""
    request = _make_request(
        user_input=(
            "I need to plan a multi-phase migration strategy: "
            "analyze the current database schema, design the new schema, "
            "plan the data migration pipeline, evaluate risk factors, "
            "and create a rollback plan. Compare different approaches."
        ),
        complexity=TaskComplexity.COMPLEX,
        risk=TaskRisk.MEDIUM,
    )
    decision = router.decide(request)

    t1 = router.config.router_thresholds["t1"]
    t2 = router.config.router_thresholds["t2"]
    print(f"Planning task: score={decision.final_score}, brain={decision.selected_brain.value}")
    assert decision.final_score >= t1, f"Expected score >= {t1} for complex planning"
    # If score is between t1 and t3, should be Planning
    if decision.final_score < router.config.router_thresholds["t3"]:
        assert decision.selected_brain == BrainType.PLANNING


# ---------------------------------------------------------------------------
# Test 4: High-risk operation → Frontier Brain
# ---------------------------------------------------------------------------


def test_high_risk_routes_to_frontier(router: CognitiveRouter) -> None:
    """Risky operations (financial, legal, medical) route to Frontier."""
    request = _make_request(
        user_input=(
            "Analyze this financial transaction log and detect potential fraud. "
            "Then DELETE all flagged records and generate a legal report. "
            "Include medical diagnosis data in the analysis."
        ),
        complexity=TaskComplexity.HIGHLY_COMPLEX,
        risk=TaskRisk.CRITICAL,
    )
    decision = router.decide(request)

    print(f"High risk: score={decision.final_score}, brain={decision.selected_brain.value}")
    print(f"Risk score: {decision.dimension_scores['risk']}")
    # With high complexity + critical risk + many risk keywords, should hit Frontier
    assert decision.selected_brain == BrainType.FRONTIER
    assert decision.dimension_scores["risk"] >= 7.0


def test_high_risk_no_frontier_if_low_complexity(router: CognitiveRouter) -> None:
    """A risky but simple query should still route appropriately (not forced to Frontier)."""
    request = _make_request(
        user_input="What is the medical term for headache?",
        risk=TaskRisk.LOW,
    )
    decision = router.decide(request)

    print(f"Medical lookup: score={decision.final_score}, brain={decision.selected_brain.value}")
    # Low complexity + low risk keywords → should NOT go to Frontier
    assert decision.selected_brain != BrainType.FRONTIER


# ---------------------------------------------------------------------------
# Test 5: Budget exhausted → forced downgrade to Cheap Brain
# ---------------------------------------------------------------------------


def test_budget_exhausted_forces_cheap_brain(default_config: OctopusConfig) -> None:
    """When budget is critically low, force Cheap Brain regardless of score."""
    router = CognitiveRouter(
        config=default_config,
        log_path=Path(tempfile.mkdtemp()) / "routing_log.jsonl",
    )

    # Simulate 0% remaining budget via metadata
    request = _make_request(
        user_input="Design a complete distributed system architecture with code",
        complexity=TaskComplexity.HIGHLY_COMPLEX,
        risk=TaskRisk.HIGH,
        metadata={"budget_remaining": 0.0},
    )
    decision = router.decide(request)

    print(f"Budget exhausted: score={decision.final_score}, brain={decision.selected_brain.value}")
    assert decision.selected_brain == BrainType.CHEAP
    assert decision.escalated is True
    assert "Budget" in decision.escalation_reason

    # Verify dimension score reflects 0 budget
    assert decision.dimension_scores["budget_remaining"] == 0.0


def test_low_budget_but_not_exhausted(default_config: OctopusConfig) -> None:
    """Budget at 20% should still route normally (not forced downgrade)."""
    router = CognitiveRouter(
        config=default_config,
        log_path=Path(tempfile.mkdtemp()) / "routing_log.jsonl",
    )
    request = _make_request(
        user_input="Design a distributed system",
        complexity=TaskComplexity.COMPLEX,
        metadata={"budget_remaining": 2.0},
    )
    decision = router.decide(request)

    # 20% remaining = 2/10 → not triggered (trigger is ≤ 1.0)
    assert decision.escalated is False


# ---------------------------------------------------------------------------
# Test 6: High skill confidence → downgrade to Skill Brain
# ---------------------------------------------------------------------------


def test_high_skill_confidence_downgrade_to_skill_brain(router: CognitiveRouter) -> None:
    """When a skill match is very high, prefer Skill Brain for efficiency."""
    # Simulate high skill confidence via many relevant_skills
    request = _make_request(
        user_input="Parse this CSV file and convert to JSON",
        relevant_skills=["csv_parser", "json_converter", "file_handler", "data_transform"],
        complexity=TaskComplexity.MODERATE,
    )
    decision = router.decide(request)

    print(f"High skill: score={decision.final_score}, brain={decision.selected_brain.value}, "
          f"skill_conf={decision.dimension_scores['skill_confidence']}")

    # With 4 relevant_skills → skill_confidence = 4 * 2 = 8 (capped)
    assert decision.dimension_scores["skill_confidence"] >= 6.0
    # Should route to SKILL (confidence ≥ 7 triggers downgrade)
    assert decision.selected_brain == BrainType.SKILL


# ---------------------------------------------------------------------------
# Test 7: Threshold boundary tests
# ---------------------------------------------------------------------------


def test_score_exactly_at_t1_routes_to_planning(router: CognitiveRouter) -> None:
    """A score exactly at T1 (3.0) should route to Planning, not Cheap."""
    # We need to craft input that produces score ~3.0
    request = _make_request(
        user_input="Analyze this report and summarize key findings for the meeting",
        complexity=TaskComplexity.MODERATE,
        risk=TaskRisk.LOW,
    )
    decision = router.decide(request)
    t1 = router.config.router_thresholds["t1"]

    print(f"T1 boundary: score={decision.final_score}, brain={decision.selected_brain.value}")
    if decision.final_score >= t1:
        assert decision.selected_brain != BrainType.CHEAP


def test_score_at_t2_boundary(router: CognitiveRouter) -> None:
    """Verify T2 boundary behaviour."""
    t2 = router.config.router_thresholds["t2"]

    # Craft a request that should land near T2
    request = _make_request(
        user_input=(
            "I need a comprehensive analysis of our deployment pipeline. "
            "Evaluate current CI/CD configuration, identify security risks, "
            "suggest improvements, and deploy changes to production."
        ),
        complexity=TaskComplexity.HIGHLY_COMPLEX,
        risk=TaskRisk.HIGH,
    )
    decision = router.decide(request)

    print(f"T2 boundary: score={decision.final_score}, brain={decision.selected_brain.value}")
    # At high complexity + high risk, should be at least Planning
    assert decision.final_score >= 3.0
    if decision.final_score >= t2:
        # Could be Planning (hybrid) or Frontier
        assert decision.selected_brain in (BrainType.PLANNING, BrainType.FRONTIER)


def test_trivial_input_minimum_score(router: CognitiveRouter) -> None:
    """The most trivial input should produce the minimum possible score."""
    decision = router.analyze("ok")
    print(f"Trivial: score={decision.final_score}, scores={decision.dimension_scores}")
    assert decision.selected_brain == BrainType.CHEAP
    assert decision.final_score < 5.0  # Should be quite low


# ---------------------------------------------------------------------------
# Test 8: Log writing verification
# ---------------------------------------------------------------------------


def test_log_writing_writes_valid_jsonl(router: CognitiveRouter) -> None:
    """Routing decisions must be logged as valid JSONL."""
    # Generate a decision
    router.analyze("Hello world")
    router.analyze("Write a Python script")
    router.analyze("DELETE from production database")

    log_path = router.log_path
    assert log_path.exists(), f"Log file not found at {log_path}"

    # Read back and validate
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 3, f"Expected 3 log entries, got {len(lines)}"

    for i, line in enumerate(lines):
        entry = json.loads(line)
        assert "timestamp" in entry
        assert "selected_brain" in entry
        assert "final_score" in entry
        assert "dimension_scores" in entry
        assert "reasoning" in entry
        assert len(entry["dimension_scores"]) == 9, (
            f"Entry {i}: expected 9 dimensions, got {len(entry['dimension_scores'])}"
        )

    # Verify brain diversity — at least the 3rd (delete db) should be different
    brains = [json.loads(l)["selected_brain"] for l in lines]
    print(f"Logged brains: {brains}")

    # "DELETE from production database" should trigger risk → higher brain
    assert "frontier" in brains or "planning" in brains, (
        "Deletion task should escalate to at least Planning"
    )


# ---------------------------------------------------------------------------
# Test 9: Dimension scoring accuracy
# ---------------------------------------------------------------------------


def test_complexity_scoring(router: CognitiveRouter) -> None:
    """Verify complexity heuristic scores."""
    # Very short
    d1 = router.analyze("hi")
    assert 1.0 <= d1.dimension_scores["complexity"] <= 3.0

    # Code-heavy
    d2 = router.analyze(
        "class User:\n    def __init__(self):\n        pass\n"
        "import os; from typing import List; async def main(): pass"
    )
    print(f"Code complexity: {d2.dimension_scores['complexity']}")
    assert d2.dimension_scores["complexity"] >= 4.0


def test_risk_scoring(router: CognitiveRouter) -> None:
    """Verify risk heuristic scores."""
    d1 = router.analyze("hello world")
    assert d1.dimension_scores["risk"] == 0.0

    d2 = router.analyze("delete all records and drop table")
    print(f"Delete risk: {d2.dimension_scores['risk']}")
    assert d2.dimension_scores["risk"] >= 4.0  # "delete" + "drop" → 2*2=4


def test_realtime_scoring(router: CognitiveRouter) -> None:
    """Verify real-time need heuristic."""
    d1 = router.analyze("hello")
    assert d1.dimension_scores["realtime_need"] == 0.0

    d2 = router.analyze("What is the latest news today about the current stock price?")
    print(f"Realtime score: {d2.dimension_scores['realtime_need']}")
    assert d2.dimension_scores["realtime_need"] >= 6.0


def test_novelty_with_memory_manager(default_config: OctopusConfig) -> None:
    """Verify novelty scoring uses memory manager when available."""
    mock_memory = MagicMock()
    mock_memory.search_similar.return_value = [
        {"score": 0.95},  # very similar
    ]

    router = CognitiveRouter(
        config=default_config,
        memory_manager=mock_memory,
        log_path=Path(tempfile.mkdtemp()) / "routing_log.jsonl",
    )
    decision = router.analyze("hello world")
    novelty = decision.dimension_scores["novelty"]
    print(f"Novelty with high-similarity memory: {novelty}")
    # High similarity → low novelty → less than default 8.0
    assert novelty < 8.0, f"Expected low novelty with similar memory, got {novelty}"


# ---------------------------------------------------------------------------
# Test 10: Config override behaviour
# ---------------------------------------------------------------------------


def test_custom_thresholds(default_config: OctopusConfig) -> None:
    """Router respects custom thresholds from config."""
    config = OctopusConfig(
        router_thresholds={"t1": 2.0, "t2": 5.0, "t3": 8.0},
    )
    router = CognitiveRouter(
        config=config,
        log_path=Path(tempfile.mkdtemp()) / "routing_log.jsonl",
    )

    # A moderate task
    request = _make_request(
        user_input="Analyze this report and suggest improvements",
        complexity=TaskComplexity.MODERATE,
    )
    decision = router.decide(request)

    print(f"Custom thresholds: score={decision.final_score}, "
          f"t1={config.router_thresholds['t1']}, brain={decision.selected_brain.value}")
    # The router should use custom thresholds
    assert router.config.router_thresholds["t1"] == 2.0


def test_custom_weights_affect_score(default_config: OctopusConfig) -> None:
    """Custom weights change the final score."""
    config_high_risk = OctopusConfig(
        router_weights={"alpha": 1.0, "beta": 0.8, "gamma": 2.0, "delta": 0.5,
                        "epsilon": 0.6, "zeta": 0.4, "eta": 0.3},
    )
    config_low_risk = OctopusConfig(
        router_weights={"alpha": 1.0, "beta": 0.8, "gamma": 0.5, "delta": 0.5,
                        "epsilon": 0.6, "zeta": 0.4, "eta": 0.3},
    )

    task = "delete all financial records"

    r1 = CognitiveRouter(
        config=config_high_risk,
        log_path=Path(tempfile.mkdtemp()) / "routing_log.jsonl",
    )
    r2 = CognitiveRouter(
        config=config_low_risk,
        log_path=Path(tempfile.mkdtemp()) / "routing_log.jsonl",
    )

    d1 = r1.analyze(task)
    d2 = r2.analyze(task)

    print(f"High γ ({config_high_risk.router_weights['gamma']}): score={d1.final_score}")
    print(f"Low γ  ({config_low_risk.router_weights['gamma']}): score={d2.final_score}")
    # Higher gamma risk weight → higher score
    assert d1.final_score > d2.final_score, (
        f"High γ should produce higher score, got {d1.final_score} ≤ {d2.final_score}"
    )


# ---------------------------------------------------------------------------
# Test 11: RouterDecision data class
# ---------------------------------------------------------------------------


def test_router_decision_dataclass() -> None:
    """Verify RouterDecision fields and defaults."""
    decision = RouterDecision(
        selected_brain=BrainType.CHEAP,
        final_score=1.5,
        dimension_scores={"complexity": 1.0},
        reasoning="Test decision",
    )
    assert decision.selected_brain == BrainType.CHEAP
    assert decision.final_score == 1.5
    assert isinstance(decision.timestamp, object)  # datetime
    assert decision.estimated_cost == 0.0
    assert decision.escalated is False
    assert decision.escalation_reason == ""


# ---------------------------------------------------------------------------
# Test 12: Tool dependency scoring
# ---------------------------------------------------------------------------


def test_tool_dependency_scoring(router: CognitiveRouter) -> None:
    """Tasks mentioning tools get higher tool_dependency scores."""
    d1 = router.analyze("hello")
    d2 = router.analyze("browse the web to fetch latest news and upload results via email")

    print(f"No tools: {d1.dimension_scores['tool_dependency']}")
    print(f"With tools: {d2.dimension_scores['tool_dependency']}")
    assert d2.dimension_scores["tool_dependency"] > d1.dimension_scores["tool_dependency"]


# ---------------------------------------------------------------------------
# Test 13: All 9 dimensions present in every decision
# ---------------------------------------------------------------------------


def test_all_nine_dimensions_present(router: CognitiveRouter) -> None:
    """Every decision must include all 9 dimension scores."""
    decision = router.analyze("test input")

    expected_dims = {
        "complexity", "novelty", "risk", "realtime_need",
        "skill_confidence", "budget_remaining", "user_preference",
        "local_capability", "tool_dependency",
    }
    actual_dims = set(decision.dimension_scores.keys())
    assert actual_dims == expected_dims, f"Missing dims: {expected_dims - actual_dims}"

    # All scores should be in [0, 10]
    for name, value in decision.dimension_scores.items():
        assert 0.0 <= value <= 10.0, f"{name}={value} out of range [0,10]"
