"""
Frontier Brain — cloud LLM reasoning for high-value tasks.

FrontierBrain is the most capable brain in the Octopus architecture.
It delegates reasoning to external LLM APIs (OpenAI-compatible protocol
via APIManager), with automatic cost tracking and budget enforcement.

Pipeline:
    1. Cost-aware API selection (cheapest API within budget)
    2. Message assembly (system prompt + compiled context + user input)
    3. API call with exponential-backoff retry
    4. Response parsing and cost recording
    5. Graceful degradation on failure (escalate or suggest cheaper brain)
"""

from __future__ import annotations

import time
from typing import Any

from octopus.api.api_manager import APIManager
from octopus.api.cost_tracker import CostTracker
from octopus.brains.base import BaseBrain, BrainRequest, BrainResponse, BrainType
from octopus.config import APIConfig

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = (
    "You are Octopus Frontier Brain. "
    "Provide thorough, accurate responses. Be concise when possible."
)

DEFAULT_MAX_RETRIES = 2


# ── FrontierBrain ───────────────────────────────────────────────────────────


class FrontierBrain(BaseBrain):
    """Cloud LLM reasoning brain for complex, high-value tasks.

    Designed to be the **most capable** but also **most expensive** brain.
    The Cognitive Router should only select Frontier when the task complexity
    score exceeds T3 (9.0) and budget allows.

    Features:
        - Cost-aware API selection based on ``BrainRequest.budget_usd``
        - Automatic cost tracking via ``CostTracker.record_call_with_price``
        - Budget enforcement: rejects tasks exceeding ``max_per_task_usd``
        - Exponential backoff retry (max 2 retries, base delay 1.0 s)
        - Context injection from ``BrainRequest.compiled_context``
        - Timeout-aware execution via ``BrainRequest.timeout_seconds``
        - Graceful degradation: suggests cheaper brain on budget failure
    """

    brain_type = BrainType.FRONTIER

    def __init__(
        self,
        api_manager: APIManager | None = None,
        cost_tracker: CostTracker | None = None,
        config: Any = None,
    ) -> None:
        """Initialize the Frontier Brain.

        Args:
            api_manager: Shared APIManager instance for LLM calls.
            cost_tracker: Shared CostTracker for budget enforcement.
            config: Optional OctopusConfig (not required when api_manager is given).
        """
        super().__init__(config)
        self._api_manager = api_manager
        self._cost_tracker = cost_tracker

    # ── Core interface ────────────────────────────────────────────────────

    def can_handle(self, request: BrainRequest) -> bool:
        """All tasks are technically handleable by Frontier — but it's expensive.

        The Cognitive Router is responsible for deciding *when* to use Frontier.
        This method always returns ``True`` so the router has full freedom.
        """
        return True

    async def process(self, request: BrainRequest) -> BrainResponse:
        """Execute the full LLM call pipeline.

        Stages:
            1. **Select API** — pick the cheapest API that fits ``request.budget_usd``
            2. **Build messages** — system prompt + compiled context + user input
            3. **Call with retry** — ``APIManager.call_with_retry`` (max 2 retries)
            4. **Parse response** — extract content and token counts
            5. **Record cost** — ``CostTracker.record_call_with_price``
            6. **Return BrainResponse** — normalized response with metrics

        Returns:
            BrainResponse with content, cost, tokens, latency, and confidence.
            On failure, returns a degraded response with ``should_escalate=True``
            or a suggestion to use a cheaper brain.
        """
        t_start = time.perf_counter()
        errors: list[str] = []

        # ── Guard: require API manager ───────────────────────────────────
        if self._api_manager is None:
            return BrainResponse(
                success=False,
                content="Frontier Brain is not available: no API manager configured.",
                brain_type=self.brain_type,
                confidence=0.0,
                errors=["APIManager not provided"],
            )

        # ── Stage 1: Select API ──────────────────────────────────────────
        try:
            api = self._select_api(request.budget_usd)
        except ValueError as exc:
            return BrainResponse(
                success=False,
                content=(
                    f"Budget insufficient for Frontier Brain. "
                    f"Task budget: ${request.budget_usd:.4f}. "
                    f"Error: {exc}. "
                    f"Try a cheaper brain (CHEAP, SKILL)."
                ),
                brain_type=self.brain_type,
                confidence=0.0,
                should_escalate=False,
                errors=[str(exc)],
            )

        # ── Stage 2: Build messages ──────────────────────────────────────
        messages = self._build_messages(
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            compiled_context=request.compiled_context,
            user_input=request.user_input,
        )

        # ── Stage 3: Call API with retry ─────────────────────────────────
        try:
            raw_response = await self._api_manager.call_with_retry(
                api_name=api.name,
                messages=messages,
                max_retries=DEFAULT_MAX_RETRIES,
                base_delay=1.0,
                max_tokens=min(request.max_tokens, api.max_tokens),
                timeout=request.timeout_seconds,
            )
        except Exception as exc:
            errors.append(f"API call exhausted retries: {exc}")
            return BrainResponse(
                success=False,
                content=(
                    f"Frontier Brain: all API attempts failed after "
                    f"{DEFAULT_MAX_RETRIES + 1} tries. "
                    f"Last error: {exc}"
                ),
                brain_type=self.brain_type,
                confidence=0.0,
                should_escalate=True,
                escalation_reason=str(exc),
                errors=errors,
            )

        # ── Stage 4: Parse response ──────────────────────────────────────
        content, tokens_in, tokens_out = self._parse_response(raw_response, errors)

        # ── Stage 5: Record cost ─────────────────────────────────────────
        cost_usd = 0.0
        total_tokens = tokens_in + tokens_out
        if self._cost_tracker is not None and total_tokens > 0:
            try:
                self._cost_tracker.record_call_with_price(
                    api_name=api.name,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    model=api.model,
                    price_per_1k_input=api.price_per_1k_input,
                    price_per_1k_output=api.price_per_1k_output,
                    task_id=request.task_id,
                )
                cost_usd = round(
                    (tokens_in / 1000.0) * api.price_per_1k_input
                    + (tokens_out / 1000.0) * api.price_per_1k_output,
                    8,
                )
            except Exception as exc:
                errors.append(f"Cost recording warning: {exc}")

        # ── Update brain stats ───────────────────────────────────────────
        self._total_tokens += total_tokens
        self._total_cost += cost_usd
        self._total_calls += 1

        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        return BrainResponse(
            success=bool(content),
            content=content or "(empty response)",
            brain_type=self.brain_type,
            tokens_used=total_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            confidence=0.9 if content else 0.0,
            errors=errors,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _select_api(self, budget_usd: float) -> APIConfig:
        """Pick the cheapest API whose estimated cost fits within the budget.

        Estimation assumes a worst-case response equal to ``api.max_tokens``
        with a 50/50 input/output token split.

        Args:
            budget_usd: Maximum USD allowed for this task.

        Returns:
            The cheapest :class:`APIConfig` that fits the budget.

        Raises:
            ValueError: If no configured API fits the budget.
        """
        if self._api_manager is None:
            raise ValueError("No API manager")

        apis = self._api_manager.list_apis()
        if not apis:
            raise ValueError("No APIs configured")

        def _estimate_cost(api: APIConfig) -> float:
            half_tokens = api.max_tokens / 2.0
            return (
                half_tokens * api.price_per_1k_input
                + half_tokens * api.price_per_1k_output
            ) / 1000.0

        for api in apis:
            est = _estimate_cost(api)
            if est <= budget_usd:
                return api

        # No API fits — tell caller the cheapest option
        cheapest = apis[0]
        cheapest_est = _estimate_cost(cheapest)
        raise ValueError(
            f"No API fits budget ${budget_usd:.4f}. "
            f"Cheapest API ({cheapest.name}, {cheapest.model}) "
            f"estimated cost: ${cheapest_est:.6f}"
        )

    @staticmethod
    def _build_messages(
        system_prompt: str,
        compiled_context: str,
        user_input: str,
    ) -> list[dict[str, str]]:
        """Assemble the messages array for an OpenAI-compatible chat completion.

        Structure::

            [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": "<compiled_context>"},   # optional
                {"role": "user",   "content": user_input},
            ]

        Args:
            system_prompt: The primary system prompt.
            compiled_context: Context compiled by the Context Compiler (may be empty).
            user_input: The original user task / query.

        Returns:
            List of message dicts ready for the API.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        # Inject compiled context as a second system message
        if compiled_context:
            context_msg = (
                f"Relevant context for this task:\n{compiled_context}\n\n"
                f"Use the above context to inform your response."
            )
            messages.append({"role": "system", "content": context_msg})

        messages.append({"role": "user", "content": user_input})
        return messages

    @staticmethod
    def _parse_response(
        raw: dict[str, Any],
        errors: list[str],
    ) -> tuple[str, int, int]:
        """Extract content and token counts from an OpenAI-compatible response.

        Args:
            raw: The full JSON response from the API.
            errors: Mutable error list for non-fatal warnings.

        Returns:
            ``(content, prompt_tokens, completion_tokens)`` triple.
            ``content`` is an empty string if parsing failed.
        """
        content = ""
        tokens_in = 0
        tokens_out = 0

        try:
            choices = raw.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "") or ""

                # Handle tool_calls if present (e.g. function-calling models)
                if "tool_calls" in msg and not content:
                    content = str(msg["tool_calls"])

            usage = raw.get("usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)

        except Exception as exc:
            errors.append(f"Response parsing warning: {exc}")

        return content, tokens_in, tokens_out
