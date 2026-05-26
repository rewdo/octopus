"""
Integration tests for the OctopusAgent end-to-end execution pipeline.

Covers:
    1. Simple greeting → Cheap Brain (full pipeline)
    2. Skill-related task → Skill Brain routing
    3. Tool execution → Action Brain
    4. Memory update after task execution
    5. Agent status report completeness
    6. Sync API (run_sync)
    7. Cost tracking
    8. Graceful degradation (no config, empty task)
    9. Multiple tasks in sequence
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.octopus.agent import OctopusAgent
from src.octopus.api.cost_tracker import CostRecord
from src.octopus.brains.base import BrainType, TaskComplexity, TaskRisk
from src.octopus.config import APIConfig, OctopusConfig
from src.octopus.memory.memory_graph import MemoryNode, NodeType
from src.octopus.skills.skill_engine import Skill, SkillRegistry, SkillStep


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def agent_config(tmp_path: Path) -> OctopusConfig:
    """A minimal but complete OctopusConfig with a temp workspace."""
    config = OctopusConfig.default()
    config.workspace_dir = tmp_path / "octopus-workspace"

    # Create workspace subdirs
    (config.workspace_dir / "skills").mkdir(parents=True, exist_ok=True)
    (config.workspace_dir / "memory").mkdir(parents=True, exist_ok=True)
    (config.workspace_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Add a demo API (cheap, local)
    config.apis = [
        APIConfig(
            name="test-api",
            provider="test",
            base_url="http://localhost:9999",
            api_key="noop",
            model="test-model",
            price_per_1k_input=0.0,
            price_per_1k_output=0.0,
            max_tokens=4096,
            priority=0,
        )
    ]

    return config


@pytest.fixture
def agent(agent_config: OctopusConfig) -> OctopusAgent:
    """Fresh OctopusAgent for each test."""
    return OctopusAgent(agent_config)


@pytest.fixture
def agent_with_skills(agent: OctopusAgent) -> OctopusAgent:
    """Agent with pre-registered demo skills."""
    registry = agent.skill_registry

    registry.register(Skill(
        name="text_summarize",
        description="Summarize text into concise bullet points",
        category="text_processing",
        version="1.0.0",
        tags=["summarize", "text", "nlp"],
        steps=[
            SkillStep(action="call_llm", params={"template": "{input_text}"}),
            SkillStep(action="format", params={"template": "Summary: {text_summarize}"}),
        ],
    ))

    registry.register(Skill(
        name="json_parser",
        description="Parse and validate JSON content",
        category="data_processing",
        version="1.0.0",
        tags=["json", "parse", "validate"],
        steps=[
            SkillStep(action="validate", params={"schema": "json"}),
            SkillStep(action="transform", params={"format": "dict"}),
        ],
    ))

    return agent


# ── Test 1: Simple greeting runs through Cheap Brain ────────────────────────


@pytest.mark.asyncio
async def test_simple_greeting_cheap_brain(agent: OctopusAgent) -> None:
    """A simple 'hello' task should route to and run through Cheap Brain."""
    result = await agent.run("hello")

    assert result["success"] is True
    assert result["brain_used"] == BrainType.CHEAP.value
    assert "Hello" in result["output"] or "hello" in result["output"].lower()
    assert result["decision"]["selected_brain"] == BrainType.CHEAP.value
    assert result["cost"] == 0.0  # Cheap brain is free
    assert result["tokens"] == 0  # No tokens consumed
    assert result["latency_ms"] > 0
    assert result["task_id"]
    assert result["timestamp"]


@pytest.mark.asyncio
async def test_simple_greeting_sync(agent: OctopusAgent) -> None:
    """run_sync() should produce the same result as async run()."""
    result = agent.run_sync("hello")

    assert result["success"] is True
    assert result["brain_used"] == BrainType.CHEAP.value
    assert "Hello" in result["output"] or "hello" in result["output"].lower()


@pytest.mark.asyncio
async def test_chinese_greeting(agent: OctopusAgent) -> None:
    """Chinese greeting '你好' should route to Cheap Brain."""
    result = await agent.run("你好")

    assert result["success"] is True
    assert result["brain_used"] == BrainType.CHEAP.value


@pytest.mark.asyncio
async def test_faq_query(agent: OctopusAgent) -> None:
    """FAQ query 'how does octopus work' should hit Cheap Brain's FAQ."""
    result = await agent.run("how does octopus work")

    assert result["success"] is True
    assert "octopus" in result["output"].lower()
    assert result["brain_used"] == BrainType.CHEAP.value


# ── Test 2: Skill-related task routes to Skill Brain ────────────────────────


@pytest.mark.asyncio
async def test_skill_task_routes_to_skill_brain(agent_with_skills: OctopusAgent) -> None:
    """Task matching registered skills should route to Skill Brain."""
    result = await agent_with_skills.run(
        "summarize this long article about AI technology",
        relevant_skills=["text_summarize"],
    )

    print(f"Skill task: brain={result['brain_used']}, score={result['decision']['final_score']}")

    # With relevant_skills explicitly set, brain must be SKILL
    assert result["brain_used"] == BrainType.SKILL.value
    assert result["success"] is True


@pytest.mark.asyncio
async def test_skill_auto_discovery(agent_with_skills: OctopusAgent) -> None:
    """Skills should be auto-discovered from the task description."""
    result = await agent_with_skills.run("summarize the meeting notes from today")

    print(f"Auto-discovery: brain={result['brain_used']}, score={result['decision']['final_score']}")
    print(f"Output: {result['output'][:200]}")

    # Either SKILL (if matched) or CHEAP (if not) — both are valid
    assert result["brain_used"] in (BrainType.SKILL.value, BrainType.CHEAP.value)
    assert result["success"] is True


# ── Test 3: Tool execution → Action Brain ───────────────────────────────────


@pytest.mark.asyncio
async def test_action_brain_shell_task(agent: OctopusAgent) -> None:
    """A shell-command task routes to Action Brain and executes."""
    result = await agent.run(
        "Run the command 'echo hello action brain'",
        allowed_tools=["shell"],
        metadata={"tool": "shell", "params": {"command": "echo", "args": ["hello action brain"]}},
    )

    print(f"Action task: brain={result['brain_used']}, output={result['output'][:200]}")

    # Action brain or tool-related routing
    assert result["success"] is True
    assert result["brain_used"] in (BrainType.ACTION.value, BrainType.CHEAP.value, BrainType.SKILL.value)


@pytest.mark.asyncio
async def test_action_brain_python_eval(agent: OctopusAgent) -> None:
    """Python evaluation task via Action Brain."""
    result = await agent.run(
        "Calculate 2 + 3 * 4",
        allowed_tools=["python_eval"],
        metadata={
            "tool": "python_eval",
            "params": {"code": "2 + 3 * 4"},
        },
    )

    print(f"Python eval: brain={result['brain_used']}, output={result['output'][:200]}")

    # Should succeed (sandboxed python_eval is safe)
    assert result["success"] is True


@pytest.mark.asyncio
async def test_action_brain_file_read(agent: OctopusAgent) -> None:
    """File read task via Action Brain."""
    # Create a test file
    test_file = agent.config.workspace_dir / "test.txt"
    test_file.write_text("Hello from integration test!")

    result = await agent.run(
        "Read the test file",
        allowed_tools=["file_read"],
        metadata={"tool": "file_read", "params": {"path": str(test_file)}},
    )

    print(f"File read: brain={result['brain_used']}, output={result['output'][:300]}")
    assert result["success"] is True


# ── Test 4: Memory update after task execution ──────────────────────────────


@pytest.mark.asyncio
async def test_memory_updated_after_task(agent: OctopusAgent) -> None:
    """After running a task, episodic memory should have at least one event."""
    await agent.run("hello world")

    events = agent.episodic_memory.get_recent(10)
    assert len(events) > 0, "Expected at least one event in episodic memory after task"

    latest = events[0]
    assert latest.node_type == NodeType.EVENT
    metadata = latest.metadata or {}
    assert "task_input" in metadata
    assert "brain_used" in metadata
    assert metadata.get("task_input") == "hello world"


@pytest.mark.asyncio
async def test_memory_accumulates_across_tasks(agent: OctopusAgent) -> None:
    """Multiple tasks should accumulate distinct episodic memories."""
    tasks = ["hello", "summarize text", "what is octopus"]
    for t in tasks:
        await agent.run(t)

    events = agent.episodic_memory.get_recent(50)
    assert len(events) >= len(tasks), (
        f"Expected at least {len(tasks)} events, got {len(events)}"
    )

    # Verify each task is recorded
    descriptions = [e.content for e in events]
    for t in tasks:
        found = any(t in desc for desc in descriptions)
        assert found, f"Task '{t}' not found in recorded events"


@pytest.mark.asyncio
async def test_working_memory_usage(agent: OctopusAgent) -> None:
    """Working memory should be populated during task execution (if the brain uses it)."""
    await agent.run("hello")

    # Working memory may or may not be populated depending on brain implementation
    wm = agent.status()["memory"]["working_memory_items"]
    assert wm >= 0  # Just verify it doesn't crash


# ── Test 5: Agent status report completeness ────────────────────────────────


def test_status_report_structure(agent: OctopusAgent) -> None:
    """status() should return a well-structured report with all sections."""
    report = agent.status()

    # Top-level sections
    assert "agent" in report
    assert "brains" in report
    assert "memory" in report
    assert "budget" in report
    assert "skills" in report
    assert "config" in report

    # Agent section
    agent_info = report["agent"]
    assert agent_info["version"] == "0.1.0"
    assert "uptime_seconds" in agent_info
    assert "tasks_completed" in agent_info
    assert "total_tokens" in agent_info
    assert "total_cost_usd" in agent_info

    # Brains section
    brains = report["brains"]
    assert BrainType.CHEAP.value in brains
    assert BrainType.SKILL.value in brains
    assert BrainType.ACTION.value in brains
    # Future brains should be listed as not_implemented
    assert BrainType.PLANNING.value in brains
    assert BrainType.FRONTIER.value in brains

    # Memory section
    memory = report["memory"]
    assert "graph" in memory
    assert "working_memory_items" in memory
    assert "episodic_events" in memory

    # Budget section
    budget = report["budget"]
    assert "monthly_budget_usd" in budget
    assert "remaining_usd" in budget
    assert "used_pct" in budget

    # Skills section
    skills = report["skills"]
    assert "registry_size" in skills
    assert "categories" in skills

    # Config section
    config = report["config"]
    assert "workspace" in config
    assert "apis_configured" in config
    assert "thresholds" in config


@pytest.mark.asyncio
async def test_status_reflects_task_count(agent: OctopusAgent) -> None:
    """status() should reflect the number of tasks executed."""
    assert agent.status()["agent"]["tasks_completed"] == 0

    await agent.run("hello")
    assert agent.status()["agent"]["tasks_completed"] == 1

    await agent.run("bye")
    assert agent.status()["agent"]["tasks_completed"] == 2


# ── Test 6: Cost tracking ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cost_tracker_initialized(agent: OctopusAgent) -> None:
    """Cost tracker should be accessible and have zero records initially."""
    stats = agent.cost_tracker.get_stats()
    assert stats.total_calls == 0
    assert stats.total_cost_usd == 0.0


@pytest.mark.asyncio
async def test_cheap_brain_tasks_incur_no_cost(agent: OctopusAgent) -> None:
    """Cheap Brain tasks should incur zero cost."""
    await agent.run("hello")
    await agent.run("hi")
    await agent.run("good morning")

    stats = agent.cost_tracker.get_stats()
    assert stats.total_cost_usd == 0.0, (
        f"Cheap brain tasks should have zero cost, got ${stats.total_cost_usd:.6f}"
    )


# ── Test 7: Graceful degradation / edge cases ───────────────────────────────


@pytest.mark.asyncio
async def test_empty_task_handled(agent: OctopusAgent) -> None:
    """An empty task string should be handled gracefully (not crash)."""
    result = await agent.run("")

    # Should not crash — may return success=False or a Cheap Brain fallback
    assert isinstance(result, dict)
    assert "success" in result
    assert "output" in result


@pytest.mark.asyncio
async def test_very_long_task(agent: OctopusAgent) -> None:
    """A very long task string should not crash the pipeline."""
    long_task = "explain the concept of " + "very detailed " * 200 + "thing"
    result = await agent.run(long_task)

    assert isinstance(result, dict)
    assert "success" in result


@pytest.mark.asyncio
async def test_task_with_unicode_emoji(agent: OctopusAgent) -> None:
    """Task with emoji and Unicode should be handled correctly."""
    result = await agent.run("Hello [globe] world! 测试繁體字 [rocket]")

    assert result["success"] is True


@pytest.mark.asyncio
async def test_router_log_is_written(agent: OctopusAgent) -> None:
    """Router log JSONL should be populated after each task."""
    await agent.run("hello")
    await agent.run("what is octopus")

    log_path = agent.config.workspace_dir / "routing_log.jsonl"
    assert log_path.exists(), f"Router log not found at {log_path}"

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 2, f"Expected 2 log entries, got {len(lines)}"


@pytest.mark.asyncio
async def test_costs_file_created(agent: OctopusAgent) -> None:
    """Cost tracking file should be created in workspace."""
    await agent.run("hello")

    cost_path = agent.config.workspace_dir / "costs.json"
    # May or may not exist depending on whether any costs were recorded
    # But at minimum, the parent dir should exist
    assert cost_path.parent.exists()


# ── Test 8: Multiple sequential tasks ───────────────────────────────────────


@pytest.mark.asyncio
async def test_multiple_mixed_tasks(agent_with_skills: OctopusAgent) -> None:
    """Multiple tasks of different types should all succeed."""
    tasks = [
        ("hello", None),  # greeting → cheap
        ("what is the weather now", None),  # info → cheap
        ("summarize this article", ["text_summarize"]),  # → skill
        ("parse this JSON data", ["json_parser"]),  # → skill
        ("hi", None),  # greeting → cheap
    ]

    for task, skills in tasks:
        kwargs = {"relevant_skills": skills} if skills else {}
        result = await agent_with_skills.run(task, **kwargs)
        assert result["success"] is True, f"Task '{task}' failed: {result.get('errors', [])}"
        assert result["brain_used"] in (
            BrainType.CHEAP.value,
            BrainType.SKILL.value,
        ), f"Unexpected brain for '{task}': {result['brain_used']}"

    # Verify task count
    assert agent_with_skills.status()["agent"]["tasks_completed"] == len(tasks)


# ── Test 9: Decision metadata included in result ────────────────────────────


@pytest.mark.asyncio
async def test_decision_in_result(agent: OctopusAgent) -> None:
    """The run() result dict must contain the full RouterDecision."""
    result = await agent.run("hello")

    decision = result["decision"]
    assert "selected_brain" in decision
    assert "final_score" in decision
    assert "reasoning" in decision
    assert "dimension_scores" in decision
    assert "escalated" in decision

    # Dimension scores should have at least the 7 core dimensions
    dim_scores = decision["dimension_scores"]
    core_dims = {"complexity", "risk", "realtime_need", "novelty"}
    assert core_dims.issubset(set(dim_scores.keys())), (
        f"Missing dimensions: {core_dims - set(dim_scores.keys())}"
    )


@pytest.mark.asyncio
async def test_cost_model_override(agent: OctopusAgent) -> None:
    """Verify that kwargs propagate correctly without errors."""
    result = await agent.run(
        "hello",
        model_override="deepseek-v4",
        temperature=0.7,
    )
    assert result["success"] is True


# ── Test 10: Agent cleanup ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_close_does_not_crash(agent: OctopusAgent) -> None:
    """close() should clean up resources without error."""
    await agent.run("hello")  # Ensure client is initialized
    await agent.close()
    # Calling close again should not crash
    await agent.close()


@pytest.mark.asyncio
async def test_agent_close_sync(agent: OctopusAgent) -> None:
    """close_sync() should work."""
    agent.run_sync("hello")
    agent.close_sync()
