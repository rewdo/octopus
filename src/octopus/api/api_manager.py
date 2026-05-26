"""Unified API manager for multi-provider LLM calls.

Supports OpenAI-compatible protocol (/v1/chat/completions) with
automatic retry, health checks, and cost-aware API selection.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional

import httpx

from octopus.config import APIConfig, OctopusConfig


class APIError(Exception):
    """Raised when an API call fails after all retries."""


class APIManager:
    """Manages multiple API endpoints with pricing-aware selection.

    Features:
    - Multi-provider support (OpenAI-compatible /v1/chat/completions)
    - Automatic sorting by unit price (cheapest first)
    - Built-in retry with exponential backoff
    - Health check with timeout
    - Environment variable resolution for API keys
    """

    def __init__(self, config: OctopusConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        # Health check cache: api_name → (is_healthy, timestamp)
        self._health_cache: dict[str, tuple[bool, float]] = {}
        self._health_cache_ttl: float = 60.0  # seconds

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-init the shared httpx client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── API listing & selection ──────────────────────────────────────────

    def list_apis(self) -> list[APIConfig]:
        """Return all configured APIs sorted by price (cheapest first).

        Sorting key: (price_per_1k_input + price_per_1k_output, priority)
        Lower price comes first; priority breaks ties.
        """
        def sort_key(api: APIConfig) -> tuple[float, int]:
            return (api.price_per_1k_input + api.price_per_1k_output, api.priority)

        return sorted(self.config.apis, key=sort_key)

    def get_api(self, name: str) -> APIConfig:
        """Get a specific API config by name.

        Raises ValueError if not found.
        """
        for api in self.config.apis:
            if api.name == name:
                return api
        raise ValueError(f"API '{name}' not found in configuration")

    def get_cheapest(self) -> APIConfig:
        """Get the cheapest available API."""
        if not self.config.apis:
            raise ValueError("No APIs configured")
        return self.list_apis()[0]

    def get_most_capable(self) -> APIConfig:
        """Get the highest-capability API (largest max_tokens wins).

        If multiple have the same max_tokens, pick the one with higher priority.
        """
        if not self.config.apis:
            raise ValueError("No APIs configured")

        def sort_key(api: APIConfig) -> tuple[int, int]:
            return (api.max_tokens, -api.priority)

        return sorted(self.config.apis, key=sort_key, reverse=True)[0]

    # ── API key resolution ───────────────────────────────────────────────

    @staticmethod
    def resolve_api_key(config: APIConfig) -> str:
        """Resolve API key, supporting $ENV_VAR references."""
        return config.resolve_api_key()

    # ── API call ─────────────────────────────────────────────────────────

    async def call(
        self,
        api_name: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make a single API call (no retry logic).

        Args:
            api_name: Name of the configured API to use.
            messages: List of message dicts in OpenAI format.
            **kwargs: Extra parameters passed to the API (temperature, max_tokens, etc.)

        Returns:
            Raw API response dict.
        """
        api = self.get_api(api_name)
        url = f"{api.base_url.rstrip('/')}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.resolve_api_key(api)}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": api.model,
            "messages": messages,
        }
        # Merge user-provided kwargs, but don't override model
        for key, value in kwargs.items():
            if key not in ("model", "api_key"):
                payload[key] = value
        # Cap max_tokens
        if "max_tokens" not in payload:
            payload["max_tokens"] = api.max_tokens

        response = await self.client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    async def call_with_retry(
        self,
        api_name: str,
        messages: list[dict[str, Any]],
        max_retries: int = 3,
        base_delay: float = 1.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call API with exponential backoff retry.

        Args:
            api_name: Name of the configured API to use.
            messages: List of message dicts in OpenAI format.
            max_retries: Maximum number of retry attempts (total calls = 1 + max_retries).
            base_delay: Initial delay in seconds (doubles each retry).
            **kwargs: Extra parameters passed to the API.

        Returns:
            Raw API response dict.

        Raises:
            APIError: If all attempts fail.
        """
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                return await self.call(api_name, messages, **kwargs)
            except httpx.HTTPStatusError as e:
                last_error = e
                # Don't retry on 4xx (client errors) except 429 (rate limit)
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise APIError(f"API call failed with {e.response.status_code}: {e}") from e

                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)

        raise APIError(f"API call failed after {max_retries + 1} attempts") from last_error

    # ── Health check ─────────────────────────────────────────────────────

    async def check_health(self, api_name: str, timeout: float = 10.0) -> bool:
        """Check if an API endpoint is healthy.

        Uses a short-lived client with a tight timeout.
        Results are cached for self._health_cache_ttl seconds.
        """
        # Check cache
        cached = self._health_cache.get(api_name)
        if cached is not None:
            is_healthy, ts = cached
            if time.time() - ts < self._health_cache_ttl:
                return is_healthy

        api = self.get_api(api_name)
        url = f"{api.base_url.rstrip('/')}/v1/models"
        headers = {"Authorization": f"Bearer {self.resolve_api_key(api)}"}

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                resp = await client.get(url, headers=headers)
                is_healthy = resp.status_code < 500
        except Exception:
            is_healthy = False

        self._health_cache[api_name] = (is_healthy, time.time())
        return is_healthy

    # ── Batch call ───────────────────────────────────────────────────────

    async def batch_call(
        self,
        api_name: str,
        messages_batch: list[list[dict[str, Any]]],
        max_concurrent: int = 3,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run multiple API calls concurrently with a semaphore.

        Args:
            api_name: Name of the configured API to use.
            messages_batch: List of message lists, one per call.
            max_concurrent: Max concurrent calls.
            **kwargs: Extra parameters passed to each API call.

        Returns:
            List of raw API response dicts (order preserved).
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_call(messages: list[dict[str, Any]]) -> dict[str, Any]:
            async with semaphore:
                return await self.call_with_retry(api_name, messages, **kwargs)

        return await asyncio.gather(*[bounded_call(msgs) for msgs in messages_batch])
