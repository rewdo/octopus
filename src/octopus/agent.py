"""
OctopusAgent — main orchestrator for the multi-brain agent.

OctopusAgent ties together the Cognitive Router, all brain instances,
memory layers, skill registry, API manager, and cost tracker into a
single end-to-end execution pipeline.

Usage::

    config = OctopusConfig.default()
    agent = OctopusAgent(config)

    # Async
    result = await agent.run("summarize this text")

    # Sync
    result = agent.run_sync("hello world")

    # Status
    print(agent.status())
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from octopus.api.api_manager import APIManager
from octopus.api.cost_tracker import CostTracker
from octopus.brains.action_brain import ActionBrain
from octopus.brains.base import BrainRequest, BrainResponse, BrainType
from octopus.brains.cheap_brain import CheapBrain
from octopus.brains.frontier_brain import FrontierBrain
from octopus.brains.memory_brain import MemoryBrain
from octopus.brains.planning_brain import PlanningBrain
from octopus.brains.skill_brain import SkillBrain
from octopus.config import OctopusConfig
from octopus.memory.context_compiler import ContextCompiler
from octopus.memory.layers import (
    EpisodicMemory,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from octopus.memory.memory_graph import MemoryGraph
from octopus.router.cognitive_router import CognitiveRouter, RouterDecision
from octopus.skills.skill_engine import SkillRegistry


# ── OctopusAgent ────────────────────────────────────────────────────────────


class OctopusAgent:
    """Main orchestrator for the Octopus multi-brain agent.

    Responsibilities:
        1. Initialize all components (brains, router, memory, skills, API, costs)
        2. Execute the end-to-end pipeline: route → compile context → execute
        3. Record costs and update memory after each task
        4. Provide status and health-check reporting
    """

    def __init__(self, config: OctopusConfig) -> None:
        """Initialize the full agent stack from configuration.

        Args:
            config: OctopusConfig with APIs, budgets, brain settings, etc.
        """
        self.config = config

        # ── Memory system ──
        self._memory_graph = MemoryGraph(graph_backend=config.memory.graph_backend)
        self._working_memory = WorkingMemory(max_items=config.memory.working_memory_size)
        self._episodic_memory = EpisodicMemory(memory_graph=self._memory_graph)
        self._semantic_memory = SemanticMemory(memory_graph=self._memory_graph)
        self._procedural_memory = ProceduralMemory(memory_graph=self._memory_graph)
        self._context_compiler = ContextCompiler()

        # ── Skill registry ──
        self._skill_registry = SkillRegistry()

        # ── API & cost tracking ──
        self._api_manager = APIManager(config)
        cost_path = config.workspace_dir / "costs.json"
        cost_path.parent.mkdir(parents=True, exist_ok=True)
        self._cost_tracker = CostTracker(
            budget=config.budget,
            storage_path=cost_path,
        )

        # ── Brain instances ──
        self._cheap_brain = CheapBrain(config=config)
        self._skill_brain = SkillBrain(registry=self._skill_registry, config=config)
        self._action_brain = ActionBrain(
            workspace_dir=config.workspace_dir,
            config=config,
        )
        self._frontier_brain = FrontierBrain(
            api_manager=self._api_manager,
            cost_tracker=self._cost_tracker,
            config=config,
        )

        self._memory_brain = MemoryBrain(
            memory_graph=self._memory_graph,
            episodic=self._episodic_memory,
            semantic=self._semantic_memory,
            config=config,
        )

        self._planning_brain = PlanningBrain(
            config=config,
        )

        # ── Brain type → instance mapping (for dispatch) ──
        self._brain_map: dict[BrainType, Any] = {
            BrainType.CHEAP: self._cheap_brain,
            BrainType.SKILL: self._skill_brain,
            BrainType.ACTION: self._action_brain,
            BrainType.MEMORY: self._memory_brain,
            BrainType.PLANNING: self._planning_brain,
            BrainType.FRONTIER: self._frontier_brain,
        }

        # ── Router ──
        self._router = CognitiveRouter(
            config=config,
            memory_manager=self._memory_graph,
            skill_library=self._skill_registry,
            budget_tracker=self._cost_tracker,
            log_path=config.workspace_dir / "routing_log.jsonl",
        )

        # ── Load skills from workspace ──
        skills_dir = config.workspace_dir / "skills"
        if skills_dir.exists():
            loaded = self._skill_registry.load_from_dir(skills_dir)
            if loaded > 0:
                import logging
                logging.debug(f"Loaded {loaded} skills from {skills_dir}")

        # ── Stats ──
        self._task_count: int = 0
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._start_time: float = time.time()

    # ── Public API ──────────────────────────────────────────────────────────

    async def run(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """Execute the full end-to-end pipeline for a task.

        Pipeline stages:
            1. Cognitive Router analyzes the task → RouterDecision
            2. Context Compiler builds relevant context from memory
            3. BrainRequest is created with compiled context
            4. Task is dispatched to the selected brain → BrainResponse
            5. Cost is recorded (if any API calls were made)
            6. Episodic memory is updated with the task result
            7. Structured result dict is returned

        Args:
            task: Natural-language task description.
            **kwargs: Optional overrides:
                - task_id: Custom task ID (auto-generated if omitted)
                - complexity: TaskComplexity override
                - risk: TaskRisk override
                - metadata: Extra metadata dict
                - relevant_skills: Pre-specified skill list
                - allowed_tools: Pre-specified tool list
                - max_tokens: Token budget for the brain
                - budget_usd: Maximum USD for this task
                - timeout_seconds: Execution timeout

        Returns:
            Dict with keys: success, output, brain_used, cost, tokens, latency_ms,
            task_id, timestamp, decision (RouterDecision fields), errors.
        """
        t_start = time.perf_counter()
        task_id = kwargs.pop("task_id", uuid.uuid4().hex[:12])
        errors: list[str] = []

        # ── Stage 1: Route ──────────────────────────────────────────────
        try:
            decision = self._router.analyze(task, task_id=task_id, **kwargs)
        except Exception as exc:
            errors.append(f"Routing failed: {exc}")
            # Fallback: use Cheap Brain
            decision = RouterDecision(
                selected_brain=BrainType.CHEAP,
                final_score=0.0,
                dimension_scores={},
                reasoning=f"Router error — fallback to Cheap Brain: {exc}",
                escalated=True,
                escalation_reason=str(exc),
            )

        # ── Stage 2: Compile context ────────────────────────────────────
        compiled_context = ""
        try:
            request_preview = BrainRequest(
                task_id=task_id,
                user_input=task,
                compiled_context="",
                relevant_memories=[],
                relevant_skills=kwargs.get("relevant_skills", []),
                complexity=kwargs.get("complexity"),
                risk=kwargs.get("risk"),
                novelty_score=kwargs.get("novelty_score", 0.0),
                max_tokens=kwargs.get("max_tokens", 4096),
                budget_usd=kwargs.get("budget_usd", self.config.budget.max_per_task_usd),
                timeout_seconds=kwargs.get("timeout_seconds", 30),
                allowed_tools=kwargs.get("allowed_tools", []),
                metadata=kwargs.get("metadata", {}),
            )
            compiled_context = self._context_compiler.compile(
                request=request_preview,
                memory_graph=self._memory_graph,
                token_budget=2000,
            )
        except Exception as exc:
            errors.append(f"Context compilation warning: {exc}")
            compiled_context = ""  # Non-fatal — continue without context

        # ── Stage 3: Build BrainRequest ─────────────────────────────────
        # Match relevant skills from task text if not explicitly provided
        relevant_skills = kwargs.get("relevant_skills", [])
        if not relevant_skills and self._skill_registry.count() > 0:
            matches = self._skill_registry.find(task)
            relevant_skills = [s.name for s in matches[:5]]

        brain_request = BrainRequest(
            task_id=task_id,
            user_input=task,
            compiled_context=compiled_context,
            relevant_memories=[],
            relevant_skills=relevant_skills,
            complexity=kwargs.get("complexity"),
            risk=kwargs.get("risk"),
            novelty_score=decision.dimension_scores.get("novelty", 0.0) / 10.0,
            max_tokens=kwargs.get("max_tokens", 4096),
            budget_usd=kwargs.get("budget_usd", self.config.budget.max_per_task_usd),
            timeout_seconds=kwargs.get("timeout_seconds", 30),
            allowed_tools=kwargs.get("allowed_tools", []),
            metadata={
                **(kwargs.get("metadata", {})),
                "router_decision": {
                    "selected_brain": decision.selected_brain.value,
                    "final_score": decision.final_score,
                    "reasoning": decision.reasoning,
                },
            },
        )

        # ── Stage 4: Dispatch to brain ──────────────────────────────────
        try:
            response = await self._dispatch(decision.selected_brain, brain_request)
        except Exception as exc:
            errors.append(f"Brain execution failed: {exc}")
            # Fallback response
            response = BrainResponse(
                success=False,
                content=f"Execution failed: {exc}",
                brain_type=decision.selected_brain,
                confidence=0.0,
                errors=[str(exc)],
            )

        # ── Stage 5: Record cost ────────────────────────────────────────
        if response.tokens_used > 0 and self.config.budget.track_costs:
            try:
                self._cost_tracker.record_call(
                    api_name=response.brain_type.value,
                    tokens_in=response.tokens_used // 2,
                    tokens_out=response.tokens_used - (response.tokens_used // 2),
                    model=f"octopus-{response.brain_type.value}",
                    task_id=task_id,
                )
            except Exception as exc:
                errors.append(f"Cost recording warning: {exc}")

        # ── Stage 6: Update memory ──────────────────────────────────────
        try:
            result_preview = response.content[:200] if response.content else "(empty)"
            event_desc = (
                f"Task: \"{task[:100]}\" → [{response.brain_type.value}] "
                f"{'✅' if response.success else '❌'} → {result_preview}"
            )
            self._episodic_memory.record_event(
                event_type="task_executed",
                description=event_desc,
                metadata={
                    "task_id": task_id,
                    "task_input": task,
                    "brain_used": response.brain_type.value,
                    "success": response.success,
                    "confidence": response.confidence,
                    "tokens_used": response.tokens_used,
                    "cost_usd": response.cost_usd,
                    "latency_ms": response.latency_ms,
                    "errors": response.errors,
                },
                importance=0.5 if response.success else 0.7,
                source="system",
            )
        except Exception as exc:
            errors.append(f"Memory update warning: {exc}")

        # ── Stage 7: Build result ───────────────────────────────────────
        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        self._task_count += 1
        self._total_tokens += response.tokens_used
        self._total_cost += response.cost_usd

        result: dict[str, Any] = {
            "success": response.success,
            "output": response.content,
            "brain_used": response.brain_type.value,
            "cost": response.cost_usd,
            "tokens": response.tokens_used,
            "latency_ms": latency_ms,
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "decision": {
                "selected_brain": decision.selected_brain.value,
                "final_score": decision.final_score,
                "reasoning": decision.reasoning,
                "estimated_cost": decision.estimated_cost,
                "dimension_scores": decision.dimension_scores,
                "escalated": decision.escalated,
                "escalation_reason": decision.escalation_reason,
            },
            "errors": response.errors + errors,
            "structured_output": response.structured_output,
            "tool_calls": response.tool_calls,
        }

        return result

    def run_sync(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """Synchronous wrapper around :meth:`run`.

        Uses `asyncio.run()` — not suitable for already-running event loops.

        Args:
            task: Natural-language task description.
            **kwargs: Same keyword arguments as :meth:`run`.

        Returns:
            Same result dict as :meth:`run`.
        """
        return asyncio.run(self.run(task, **kwargs))

    def status(self) -> dict[str, Any]:
        """Return comprehensive agent status report.

        Includes brain stats, memory state, budget, skills, and runtime metrics.

        Returns:
            Dict with nested stats sections.
        """
        uptime_seconds = time.time() - self._start_time
        cost_stats = self._cost_tracker.get_stats()

        brain_stats = {}
        for btype, brain in self._brain_map.items():
            if brain is not None and hasattr(brain, "stats"):
                brain_stats[btype.value] = brain.stats

        # Also include brains without instances (future)
        for btype in BrainType:
            if btype not in self._brain_map:
                brain_stats[btype.value] = {"status": "not_implemented"}

        return {
            "agent": {
                "version": "0.1.0",
                "uptime_seconds": round(uptime_seconds, 1),
                "tasks_completed": self._task_count,
                "total_tokens": self._total_tokens,
                "total_cost_usd": round(self._total_cost, 6),
            },
            "brains": brain_stats,
            "memory": {
                "graph": self._memory_graph.stats(),
                "working_memory_items": len(self._working_memory),
                "episodic_events": len(self._episodic_memory),
                "semantic_facts_prefs": len(self._semantic_memory),
                "procedural_skills": len(self._procedural_memory),
            },
            "budget": {
                "monthly_budget_usd": self.config.budget.monthly_budget_usd,
                "spent_usd": cost_stats.total_cost_usd,
                "remaining_usd": round(cost_stats.remaining_budget, 6),
                "used_pct": cost_stats.budget_used_pct,
                "calls": cost_stats.total_calls,
                "near_limit": self._cost_tracker.is_near_limit(),
            },
            "skills": {
                "registry_size": self._skill_registry.count(),
                "categories": self._skill_registry.list_categories(),
                "tags": self._skill_registry.list_tags(),
            },
            "config": {
                "workspace": str(self.config.workspace_dir),
                "apis_configured": len(self.config.apis),
                "thresholds": self.config.router_thresholds,
            },
        }

    # ── Internal methods ────────────────────────────────────────────────────

    async def _dispatch(
        self,
        brain_type: BrainType,
        request: BrainRequest,
    ) -> BrainResponse:
        """Dispatch a BrainRequest to the appropriate brain instance.

        If the requested brain is not yet implemented (e.g., Planning, Frontier),
        falls back to the most appropriate available brain.

        Args:
            brain_type: The brain selected by the router.
            request: Fully assembled BrainRequest.

        Returns:
            BrainResponse from the executing brain.
        """
        brain = self._brain_map.get(brain_type)

        if brain is None:
            # Fallback for unimplemented brains
            fallback = await self._fallback_brain(brain_type, request)
            if fallback:
                return fallback

            # Ultimate fallback: Cheap Brain
            brain = self._cheap_brain

        return await brain.process(request)

    async def _fallback_brain(
        self,
        brain_type: BrainType,
        request: BrainRequest,
    ) -> Optional[BrainResponse]:
        """Provide a fallback response when the selected brain is not available.

        Args:
            brain_type: The brain type that was requested but is not implemented.
            request: The original BrainRequest.

        Returns:
            A BrainResponse explaining the fallback, or None to use Cheap Brain.
        """
        available = ", ".join(b.value for b in self._brain_map.keys())
        msg = (
            f"[Fallback] Brain '{brain_type.value}' is not yet implemented. "
            f"Available brains: {available}. "
            f"Using Cheap Brain as ultimate fallback."
        )

        if brain_type == BrainType.PLANNING:
            # Try Skill Brain first, fallback to Cheap if skills don't match
            if self._skill_brain.can_handle(request):
                try:
                    return await self._skill_brain.process(request)
                except Exception:
                    pass
            # Fall through to ultimate cheap brain fallback
            return None
        elif brain_type == BrainType.FRONTIER:
            # Try Skill Brain, then fall to Cheap
            # For unimplemented FRONTIER, escalate to a warning
            return BrainResponse(
                success=False,
                content=(
                    f"Frontier Brain not yet available. "
                    f"Task requires high-capability reasoning that is scheduled for Phase 2."
                ),
                brain_type=BrainType.FRONTIER,
                confidence=0.0,
                should_escalate=True,
                escalation_reason="Frontier Brain not implemented (Phase 2)",
                errors=[msg],
            )
        elif brain_type in (BrainType.MEMORY, BrainType.WORLD):
            # Memory-intensive tasks can use Cheap Brain with existing context
            return None  # Let caller fall back to Cheap

        return None

    # ── Property accessors ──────────────────────────────────────────────────

    @property
    def memory_graph(self) -> MemoryGraph:
        """Access the underlying memory graph."""
        return self._memory_graph

    @property
    def episodic_memory(self) -> EpisodicMemory:
        """Access episodic memory (timeline of events)."""
        return self._episodic_memory

    @property
    def semantic_memory(self) -> SemanticMemory:
        """Access semantic memory (facts and preferences)."""
        return self._semantic_memory

    @property
    def skill_registry(self) -> SkillRegistry:
        """Access the skill registry."""
        return self._skill_registry

    @property
    def cost_tracker(self) -> CostTracker:
        """Access the cost tracker."""
        return self._cost_tracker

    @property
    def router(self) -> CognitiveRouter:
        """Access the cognitive router."""
        return self._router

    @property
    def task_count(self) -> int:
        """Number of tasks executed so far."""
        return self._task_count

    # ── Cleanup ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Clean up resources (HTTP clients, etc.)."""
        await self._api_manager.close()

    def close_sync(self) -> None:
        """Synchronous cleanup wrapper."""
        try:
            asyncio.run(self.close())
        except RuntimeError:
            # Already in an event loop — schedule it
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
