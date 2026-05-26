"""
Skill Arena — benchmark, regression test, leaderboard, and auto-retirement.

Provides continuous quality assurance for registered skills by running
benchmarks against test inputs and retiring degraded versions.

Usage::

    registry = SkillRegistry()
    registry.load_from_dir("skills/")
    arena = SkillArena(registry)

    # Single benchmark
    result = arena.benchmark("text_summarize", ["Hello world"])
    print(result.success_rate)

    # Full regression
    results = arena.run_all(SkillArena.DEFAULT_SUITE)
    for r in results:
        print(f"{r.skill_name}: {r.success_rate:.1%}")

    # Leaderboard
    for r in arena.get_leaderboard():
        print(f"{r.rank}. {r.skill_name} — {r.success_rate:.1%}")

    # Auto-retire weak skills
    axed = arena.auto_retire(threshold_success_rate=0.5)
    print(f"Retired: {axed}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .skill_engine import SkillRegistry, Skill


# ---------------------------------------------------------------------------
# Default test suite (Phase 1: built-in reference inputs)
# ---------------------------------------------------------------------------

DEFAULT_SUITE: dict[str, list[str]] = {
    "text_summarize": ["The quick brown fox jumps over the lazy dog."],
    "text_translate": ["Hello world"],
    "extract_email": ["Contact me at test@example.com"],
    "code_snippet": ["sort a list in python"],
    "web_search": ["python documentation"],
    "calc_math": ["2 + 2 * 3"],
    "uuid_generate": [""],
    "text_sentiment": ["I am very happy today!"],
    "extract_url": ["Visit us at https://example.com"],
    "extract_phone": ["Call me at +1-555-0123"],
    "text_format": ["hello WORLD"],
    "code_review": ["def add(a, b): return a + b"],
    "code_refactor": ["for i in range(10): print(i)"],
    "web_fetch": ["https://example.com"],
    "file_read": ["/tmp/test.txt"],
    "base64_encode": ["hello"],
    "datetime_convert": ["2024-01-01"],
    "json_format": ['{"key":"value"}'],
    "text_extract_keywords": ["Python is great for data science"],
    "code_explain": ["print('hi')"],
    "code_test_gen": ["def multiply(a, b): return a * b"],
    "web_check_status": ["https://example.com"],
    "file_search": ["*.py"],
    "file_convert": ["data.csv"],
}


# ---------------------------------------------------------------------------
# ArenaResult
# ---------------------------------------------------------------------------

@dataclass
class ArenaResult:
    """Result of benchmarking a single skill.

    Attributes:
        skill_name: The registered skill name.
        version: The skill version tested.
        passed: Whether the skill meets the minimum quality bar.
        success_rate: Fraction of test inputs that succeeded (0.0–1.0).
        avg_latency_ms: Average execution time in milliseconds.
        avg_cost_usd: Average estimated cost per invocation in USD.
        total_runs: Number of test inputs executed.
        failures: Log messages for each failing test input.
        rank: Position in the leaderboard (set after ranking).
    """

    skill_name: str
    version: str
    passed: bool
    success_rate: float
    avg_latency_ms: float
    avg_cost_usd: float
    total_runs: int
    failures: list[str] = field(default_factory=list)
    rank: int = 0

    def __repr__(self) -> str:
        return (
            f"ArenaResult({self.skill_name} v{self.version}, "
            f"pass={self.passed}, success={self.success_rate:.1%}, "
            f"latency={self.avg_latency_ms:.1f}ms, "
            f"cost=${self.avg_cost_usd:.6f})"
        )


# ---------------------------------------------------------------------------
# SkillArena
# ---------------------------------------------------------------------------

class SkillArena:
    """Runs benchmark evaluations against registered skills.

    Validates skills structurally and simulates execution to compute
    success rate, latency, and cost estimates.  Exposes a leaderboard
    and a safety net (auto_retire) to keep the registry healthy.

    Parameters:
        skill_registry: An active ``SkillRegistry`` with loaded skills.
    """

    # Latency multiplier applied per skill step (ms), simulating I/O bound work.
    BASE_MS_PER_STEP = 50.0
    # Extra latency for LLM-type actions (e.g., call_llm, generate).
    LLM_MS_PER_STEP = 200.0
    # Estimated cost per LLM step (USD).
    LLM_COST_PER_STEP = 0.0001
    # Default cost per non-LLM step.
    BASE_COST_PER_STEP = 0.000001

    def __init__(self, skill_registry: "SkillRegistry") -> None:
        self._registry = skill_registry

    # ── Benchmark single skill ──────────────────────────────────────────

    def benchmark(
        self, skill_name: str, test_inputs: list[str]
    ) -> ArenaResult:
        """Run a skill against a list of test inputs and return stats.

        Each test input is evaluated by simulating the skill's step
        pipeline.  Structural problems (missing steps, malformed actions)
        count as failures.

        Args:
            skill_name: Registered skill name.
            test_inputs: List of input strings to feed the skill.

        Returns:
            ArenaResult with aggregated statistics.

        Raises:
            KeyError: If the skill is not registered.
        """
        skill = self._registry.get(skill_name)
        if skill is None:
            raise KeyError(f"Skill not registered: {skill_name}")

        if not test_inputs:
            return ArenaResult(
                skill_name=skill.name,
                version=skill.version,
                passed=True,
                success_rate=1.0,
                avg_latency_ms=0.0,
                avg_cost_usd=0.0,
                total_runs=0,
            )

        total_lat = 0.0
        total_cost = 0.0
        successes = 0
        failures: list[str] = []

        for idx, test_input in enumerate(test_inputs):
            start = time.perf_counter()
            success, msg = self._simulate_execution(skill, test_input)
            elapsed = (time.perf_counter() - start) * 1000.0
            cost = self._estimate_cost(skill)

            total_lat += elapsed
            total_cost += cost

            if success:
                successes += 1
            else:
                failures.append(f"input[{idx}]='{test_input[:60]}' — {msg}")

        total = len(test_inputs)
        success_rate = successes / total
        passed = success_rate >= 0.8  # 80% is the arena quality gate

        return ArenaResult(
            skill_name=skill.name,
            version=skill.version,
            passed=passed,
            success_rate=round(success_rate, 4),
            avg_latency_ms=round(total_lat / total, 2),
            avg_cost_usd=round(total_cost / total, 6),
            total_runs=total,
            failures=failures,
        )

    # ── Run all skills in a suite ───────────────────────────────────────

    def run_all(
        self, test_suite: dict[str, list[str]]
    ) -> list[ArenaResult]:
        """Benchmark every skill named in the test suite.

        Args:
            test_suite: ``{skill_name: [test_input, ...]}`` mapping.

        Returns:
            List of ArenaResult, one per non-missing skill in the suite.
        """
        results: list[ArenaResult] = []
        for skill_name, inputs in test_suite.items():
            if not self._registry.has(skill_name):
                continue
            result = self.benchmark(skill_name, inputs)
            results.append(result)
        return results

    # ── Compare two versions ────────────────────────────────────────────

    def compare_versions(
        self, skill_name: str, v1: str, v2: str
    ) -> dict:
        """Compare the performance of two hypothetical versions.

        This is used to evaluate upgrades *before* promoting a new
        version into the live registry.  The comparison re-runs
        benchmarks using the currently registered skill for both
        versions (the registry only holds one version at a time).

        Returns a dict with keys: ``skill_name``, ``v1``, ``v2``,
        ``v1_result``, ``v2_result``, ``delta_success_rate`` (pp),
        ``delta_latency_ms``, ``delta_cost_usd``, ``winner``.

        Args:
            skill_name: Registered skill name.
            v1: First version label (string).
            v2: Second version label (string).

        Returns:
            Comparison dictionary.
        """
        skill = self._registry.get(skill_name)
        if skill is None:
            raise KeyError(f"Skill not registered: {skill_name}")

        # Re-use the stored test inputs from the default suite
        inputs = DEFAULT_SUITE.get(skill_name, [])

        # Benchmark the current version and stamp version labels
        result_v1 = self.benchmark(skill_name, inputs)
        result_v2 = self.benchmark(skill_name, inputs)

        result_v1.version = v1
        result_v2.version = v2

        delta_sr = round(result_v2.success_rate - result_v1.success_rate, 4)
        delta_lat = round(result_v2.avg_latency_ms - result_v1.avg_latency_ms, 2)
        delta_cost = round(result_v2.avg_cost_usd - result_v1.avg_cost_usd, 6)

        if delta_sr > 0:
            winner = v2
        elif delta_sr < 0:
            winner = v1
        else:
            winner = "tie"

        return {
            "skill_name": skill_name,
            "v1": v1,
            "v2": v2,
            "v1_result": result_v1,
            "v2_result": result_v2,
            "delta_success_rate": delta_sr,
            "delta_latency_ms": delta_lat,
            "delta_cost_usd": delta_cost,
            "winner": winner,
        }

    # ── Leaderboard ─────────────────────────────────────────────────────

    def get_leaderboard(
        self, sort_by: str = "success_rate"
    ) -> list[ArenaResult]:
        """Benchmark all registered skills and return a ranked leaderboard.

        Args:
            sort_by: Metric to sort on — ``"success_rate"`` (default),
                     ``"avg_latency_ms"`` (ascending), or ``"avg_cost_usd"`` (ascending).

        Returns:
            Rank-annotated ArenaResult list.
        """
        all_skills = self._registry.list_all()
        results: list[ArenaResult] = []

        for skill in all_skills:
            inputs = DEFAULT_SUITE.get(skill.name, [])
            result = self.benchmark(skill.name, inputs)
            results.append(result)

        # Sort
        reverse = sort_by == "success_rate"  # higher is better
        key_map = {
            "success_rate": lambda r: r.success_rate,
            "avg_latency_ms": lambda r: -r.avg_latency_ms,  # negate so ascending
            "avg_cost_usd": lambda r: -r.avg_cost_usd,
        }
        key_fn = key_map.get(sort_by, key_map["success_rate"])
        results.sort(key=key_fn, reverse=reverse)

        for idx, r in enumerate(results, 1):
            r.rank = idx

        return results

    # ── Auto-retire ─────────────────────────────────────────────────────

    def auto_retire(
        self, threshold_success_rate: float = 0.5
    ) -> list[str]:
        """Benchmark all registered skills and unregister those below
        the success-rate threshold.

        Args:
            threshold_success_rate: Skills with success_rate below this
                                    value are retired.

        Returns:
            List of retired skill names.
        """
        retired: list[str] = []
        for skill in self._registry.list_all():
            inputs = DEFAULT_SUITE.get(skill.name, [])
            result = self.benchmark(skill.name, inputs)
            if result.success_rate < threshold_success_rate:
                self._registry.unregister(skill.name)
                retired.append(skill.name)

        return retired

    # ── Internal: simulation helpers ────────────────────────────────────

    @staticmethod
    def _estimate_cost(skill: "Skill") -> float:
        """Estimate USD cost by counting LLM-call steps."""
        total = 0.0
        for step in skill.steps:
            if step.action in ("call_llm", "generate", "summarize", "translate"):
                total += SkillArena.LLM_COST_PER_STEP
            else:
                total += SkillArena.BASE_COST_PER_STEP
        return round(total, 8)

    @staticmethod
    def _simulate_execution(
        skill: "Skill", _test_input: str
    ) -> tuple[bool, str]:
        """Simulate executing a skill.

        Phase 1 validates structural integrity — future phases will
        wire in real LLM execution.

        Returns:
            (success, message) tuple.
        """
        if not skill.steps:
            return False, "no steps defined"

        for idx, step in enumerate(skill.steps):
            if not step.action or not isinstance(step.action, str):
                return False, f"step[{idx}]: missing or invalid action"
            if step.action.strip() == "":
                return False, f"step[{idx}]: empty action string"

        # Simulated success — structural validation passed
        return True, "ok"

    def __repr__(self) -> str:
        return (
            f"SkillArena(registry={self._registry})"
        )
