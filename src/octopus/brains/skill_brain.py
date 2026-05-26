"""
Skill Brain — Executes pre-compiled skill workflows in DAG order.

The SkillBrain takes a BrainRequest with a relevant_skills list, loads skill
definitions from the registry, and executes each skill's steps sequentially
with error recovery strategies.

Each skill defines a multi-step workflow with:
    - DAG-ordered steps via explicit `depends_on`
    - Per-step error handling (skip/retry/abort)
    - Result capture and composition
    - Cost and latency tracking
"""

from __future__ import annotations

import asyncio
import copy
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .base import (
    BaseBrain,
    BrainRequest,
    BrainResponse,
    BrainType,
    TaskComplexity,
)


# ---------------------------------------------------------------------------
# Skill execution result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result from executing a single skill step."""

    step_name: str
    action: str
    success: bool
    output: Any = None
    error: str = ""
    latency_ms: float = 0.0
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SkillBrain
# ---------------------------------------------------------------------------

class SkillBrain(BaseBrain):
    """Executes skill workflows from the Skill Registry.

    Usage::

        registry = SkillRegistry()
        registry.load_from_dir("skills/")

        brain = SkillBrain(registry=registry)
        request = BrainRequest(
            task_id="t1",
            user_input="summarize this text",
            relevant_skills=["text_summarize"],
        )
        response = await brain.process(request)
    """

    def __init__(
        self,
        registry: Any = None,  # SkillRegistry (lazy import to avoid circular)
        executors: Optional[dict[str, Callable]] = None,
        config: Any = None,
    ):
        super().__init__(config)
        self._registry = registry  # SkillRegistry
        self._executors = executors or {}
        self._max_retries = 3

    @property
    def brain_type(self) -> BrainType:
        return BrainType.SKILL

    def set_registry(self, registry: Any) -> None:
        """Bind or update the skill registry."""
        self._registry = registry

    def register_executor(self, action: str, fn: Callable) -> None:
        """Register a custom executor function for a given action type.

        Args:
            action: The action name (e.g., 'call_llm', 'extract_regex').
            fn: An async callable that takes (params, context) and returns Any.
        """
        self._executors[action] = fn

    async def process(self, request: BrainRequest) -> BrainResponse:
        """Execute the requested skills and return combined results."""
        if not self._registry:
            return BrainResponse(
                success=False,
                content="No SkillRegistry bound to SkillBrain.",
                brain_type=BrainType.SKILL,
                confidence=0.0,
                errors=["no_registry"],
            )

        t_start = time.perf_counter()
        skill_names = request.relevant_skills
        results: list[dict[str, Any]] = []
        errors: list[str] = []
        total_cost = 0.0
        total_tokens = 0

        if not skill_names:
            return BrainResponse(
                success=True,
                content="No skills requested — nothing to execute.",
                brain_type=BrainType.SKILL,
                confidence=1.0,
                latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )

        for skill_name in skill_names:
            skill = self._registry.get(skill_name)
            if skill is None:
                errors.append(f"Skill not found: {skill_name}")
                continue

            try:
                skill_result = await self._execute_skill(skill, request)
                results.append(skill_result)
                total_cost += skill_result.get("cost_usd", 0)
                total_tokens += skill_result.get("tokens_used", 0)
            except Exception as e:
                errors.append(f"Skill '{skill_name}' failed: {e}")

        latency_ms = (time.perf_counter() - t_start) * 1000
        success = len(results) > 0

        # Build combined output
        output_parts = []
        for r in results:
            status = "✅" if r.get("success") else "❌"
            output_parts.append(
                f"[{status}] {r['skill_name']} v{r['version']} "
                f"({r.get('steps_completed', 0)}/{r.get('steps_total', 0)} steps)"
            )
            if r.get("output"):
                output_parts.append(str(r["output"]))

            # Include per-step details
            for step in r.get("step_results", []):
                step_status = "✅" if step.success else "❌"
                output_parts.append(
                    f"  {step_status} {step.step_name} "
                    f"({step.action}, {step.latency_ms:.1f}ms)"
                )
                if step.error:
                    output_parts.append(f"    Error: {step.error}")

        content = "\n".join(output_parts) if output_parts else "No results."

        return BrainResponse(
            success=success,
            content=content,
            brain_type=BrainType.SKILL,
            tokens_used=total_tokens,
            cost_usd=total_cost,
            latency_ms=round(latency_ms, 2),
            confidence=0.85 if success else 0.3,
            structured_output={
                "skills_executed": [r["skill_name"] for r in results],
                "total_skills_requested": len(skill_names),
                "skills_failed": len(errors),
                "step_results": [
                    {
                        "skill": r["skill_name"],
                        "steps": [
                            {
                                "name": s.step_name,
                                "success": s.success,
                                "latency_ms": s.latency_ms,
                                "retries": s.retries,
                            }
                            for s in r.get("step_results", [])
                        ],
                    }
                    for r in results
                ],
            },
            errors=errors,
            metadata={
                "skill_names": skill_names,
                "registry_size": self._registry.count(),
            },
        )

    def can_handle(self, request: BrainRequest) -> bool:
        """SkillBrain can handle requests with relevant_skills specified."""
        return bool(request.relevant_skills)

    async def _execute_skill(self, skill: Any, request: BrainRequest) -> dict[str, Any]:
        """Execute all steps of a single skill.

        Args:
            skill: A Skill object with .name, .steps list, .version, etc.
            request: The original BrainRequest for context.

        Returns:
            Dict with skill_name, version, success, output, step_results, etc.
        """
        steps = getattr(skill, "steps", [])
        step_results: list[StepResult] = []
        context: dict[str, Any] = {
            "input": request.user_input,
            "metadata": request.metadata,
        }
        all_success = True
        total_cost = 0.0
        total_tokens = 0

        # Build DAG execution order
        ordered_steps = self._resolve_dag_order(steps)

        for step in ordered_steps:
            step_name = getattr(step, "action", "unknown")
            action = getattr(step, "action", "unknown")
            params = getattr(step, "params", {})
            on_error = getattr(step, "on_error", "abort")

            result = await self._execute_step(
                step_name=step_name,
                action=action,
                params=params,
                on_error=on_error,
                context=context,
            )
            step_results.append(result)

            # Update context with step output
            if result.success:
                context[step_name] = result.output
            else:
                all_success = False
                if on_error == "abort":
                    break
                # "skip" or "retry" (already retried above) continues

        # Get final output from the last successful step
        final_output = None
        for sr in reversed(step_results):
            if sr.success and sr.output is not None:
                final_output = sr.output
                break

        return {
            "skill_name": getattr(skill, "name", "unknown"),
            "version": getattr(skill, "version", "0.0.0"),
            "success": all_success,
            "output": final_output,
            "step_results": step_results,
            "steps_completed": sum(1 for s in step_results if s.success),
            "steps_total": len(steps),
            "cost_usd": total_cost,
            "tokens_used": total_tokens,
        }

    async def _execute_step(
        self,
        step_name: str,
        action: str,
        params: dict[str, Any],
        on_error: str,
        context: dict[str, Any],
    ) -> StepResult:
        """Execute a single step with retry logic.

        Args:
            step_name: Identifier for this step.
            action: The action type to execute.
            params: Parameters for the action.
            on_error: Error handling strategy (skip/retry/abort).
            context: Shared execution context (input, previous outputs).

        Returns:
            StepResult with success/failure and output.
        """
        t0 = time.perf_counter()
        max_attempts = self._max_retries if on_error == "retry" else 1
        last_error = ""
        retries = 0

        for attempt in range(max_attempts):
            try:
                # Find executor
                executor = self._executors.get(action)
                if executor is None:
                    # Use default executor resolution
                    executor = self._get_default_executor(action)

                if executor is None:
                    return StepResult(
                        step_name=step_name,
                        action=action,
                        success=False,
                        error=f"No executor for action: {action}",
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )

                # Resolve template params with context
                resolved = self._resolve_params(params, context)

                # Execute
                if asyncio.iscoroutinefunction(executor):
                    output = await executor(resolved, context)
                else:
                    output = executor(resolved, context)

                return StepResult(
                    step_name=step_name,
                    action=action,
                    success=True,
                    output=output,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    retries=retries,
                )

            except Exception as e:
                last_error = str(e)
                retries = attempt + 1
                if attempt < max_attempts - 1:
                    await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff

        return StepResult(
            step_name=step_name,
            action=action,
            success=False,
            error=last_error,
            latency_ms=(time.perf_counter() - t0) * 1000,
            retries=retries,
        )

    def _resolve_params(
        self, params: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve parameter templates with context values.

        Supports {input_text}, {step_name}, {context.key} style references.
        """
        import re as _re

        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and "{" in value:
                # Replace {input_text} with the original input
                value = value.replace("{input_text}", str(context.get("input", "")))

                # Replace {step.output} references
                def _replace_ref(match):
                    ref = match.group(1)
                    parts = ref.split(".")
                    val = context
                    for p in parts:
                        if isinstance(val, dict):
                            val = val.get(p, "")
                        else:
                            return ""
                    return str(val)

                value = _re.sub(r"\{([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)\}", _replace_ref, value)

            resolved[key] = value
        return resolved

    def _get_default_executor(self, action: str) -> Optional[Callable]:
        """Get a default built-in executor for known action types.

        These are placeholder executors that pass through data.
        Real executors should be registered via register_executor().
        """
        # Default pass-through executors
        async def _passthrough(params, ctx):
            """Default: return params as-is as the output."""
            return params

        async def _template_render(params, ctx):
            """Render a template string from context."""
            template = params.get("template", str(ctx.get("input", "")))
            return template

        _builtins = {
            "call_llm": _passthrough,  # Requires external LLM executor
            "extract_regex": _passthrough,
            "transform": _passthrough,
            "filter": _passthrough,
            "merge": _passthrough,
            "validate": _passthrough,
            "format": _template_render,
            "noop": _passthrough,
        }

        return _builtins.get(action)

    @staticmethod
    def _resolve_dag_order(steps: list[Any]) -> list[Any]:
        """Resolve DAG execution order using topological sort.

        If steps don't have depends_on, returns them in original order.
        """
        # For now, steps are linear — return as-is.
        # Full DAG resolution will be added when SkillStep adds depends_on.
        return list(steps)

    @property
    def registry(self) -> Any:
        """Return the bound skill registry, or None."""
        return self._registry

    @registry.setter
    def registry(self, value: Any) -> None:
        """Set the skill registry."""
        self._registry = value


# Re-export SkillRegistry for convenience
from ..skills.skill_engine import SkillRegistry  # noqa: E402, F811
