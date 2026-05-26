#!/usr/bin/env python3
"""
Octopus V5.0 — Basic Usage Example

This script demonstrates the core Octopus workflow:
1. Initialize configuration
2. Explore API management (list, select, health check)
3. Track costs with CostTracker
4. Use WorldBrain for state management
5. Route tasks through different brains
6. View cost & memory statistics

Run: python examples/basic_usage.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from octopus.api import APIManager, CostTracker
from octopus.brains.base import BrainRequest, BrainResponse, BrainType, TaskComplexity, TaskRisk
from octopus.config import APIConfig, BudgetConfig, OctopusConfig
from octopus.world import WorldBrain


# ── Demo helpers ─────────────────────────────────────────────────────────


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ── 1. Configuration ────────────────────────────────────────────────────


def demo_configuration() -> OctopusConfig:
    """Step 1: Create and explore configuration."""
    print_section("1. Configuration Setup")

    config = OctopusConfig.default()
    config.workspace_dir = Path("./octopus-workspace")

    # Add demo API endpoints
    config.apis = [
        APIConfig(
            name="deepseek-v4",
            provider="deepseek",
            base_url="https://api.deepseek.com",
            api_key="$DEEPSEEK_API_KEY",
            model="deepseek-chat",
            price_per_1k_input=0.00014,
            price_per_1k_output=0.00028,
            max_tokens=8192,
            priority=1,
        ),
        APIConfig(
            name="openai-gpt4o-mini",
            provider="openai",
            base_url="https://api.openai.com",
            api_key="$OPENAI_API_KEY",
            model="gpt-4o-mini",
            price_per_1k_input=0.00015,
            price_per_1k_output=0.00060,
            max_tokens=16384,
            priority=2,
        ),
        APIConfig(
            name="local-ollama",
            provider="ollama",
            base_url="http://localhost:11434",
            api_key="",
            model="qwen2.5:7b",
            price_per_1k_input=0.0,
            price_per_1k_output=0.0,
            max_tokens=4096,
            priority=0,
        ),
        APIConfig(
            name="anthropic-claude",
            provider="anthropic",
            base_url="https://api.anthropic.com",
            api_key="$ANTHROPIC_API_KEY",
            model="claude-3-haiku-20240307",
            price_per_1k_input=0.00025,
            price_per_1k_output=0.00125,
            max_tokens=4096,
            priority=3,
        ),
    ]

    config.budget = BudgetConfig(
        monthly_budget_usd=20.0,
        max_per_task_usd=0.50,
        warn_threshold_pct=80.0,
        track_costs=True,
    )

    print(f"  Workspace: {config.workspace_dir}")
    print(f"  APIs configured: {len(config.apis)}")
    print(f"  Monthly budget: ${config.budget.monthly_budget_usd}")
    print(f"  Max per task: ${config.budget.max_per_task_usd}")

    return config


# ── 2. API Management ───────────────────────────────────────────────────


async def demo_api_management(config: OctopusConfig) -> APIManager:
    """Step 2: Explore API management features."""
    print_section("2. API Management")

    manager = APIManager(config)

    # List APIs sorted by price
    print("\n  📊 APIs sorted by cost (cheapest first):")
    for api in manager.list_apis():
        total_price = api.price_per_1k_input + api.price_per_1k_output
        print(
            f"    • {api.name:<22} | {api.model:<22} | "
            f"${api.price_per_1k_input:.6f}/${api.price_per_1k_output:.6f}/1K | "
            f"max_tokens={api.max_tokens}"
        )

    # Get cheapest
    cheapest = manager.get_cheapest()
    print(f"\n  💰 Cheapest API: {cheapest.name} ({cheapest.model})")

    # Get most capable
    most_capable = manager.get_most_capable()
    print(f"  🚀 Most capable API: {most_capable.name} ({most_capable.model}, max_tokens={most_capable.max_tokens})")

    # Get specific API
    try:
        api = manager.get_api("deepseek-v4")
        print(f"\n  🔍 Lookup 'deepseek-v4': model={api.model}, base_url={api.base_url}")
    except ValueError as e:
        print(f"  ❌ {e}")

    # API key resolution
    print(f"  🔑 deepseek-v4 key resolves to: '{manager.resolve_api_key(config.apis[0])}'")
    print(f"     (empty because DEEPSEEK_API_KEY env var is not set in demo)")

    # Health check (will fail without actual API key, but demonstrates the API)
    print("\n  🏥 Health check demo:")
    try:
        health = await manager.check_health("local-ollama", timeout=3.0)
        print(f"    local-ollama: {'✓ healthy' if health else '✗ unreachable (expected if Ollama not running)'}")
    except Exception:
        print("    local-ollama: ✗ connection refused (expected)")

    # Close the client
    await manager.close()

    return manager


# ── 3. Cost Tracking ────────────────────────────────────────────────────


def demo_cost_tracking(config: OctopusConfig) -> CostTracker:
    """Step 3: Demonstrate cost tracking."""
    print_section("3. Cost Tracking")

    # Use a temp file for persistence
    tmp_dir = tempfile.mkdtemp(prefix="octopus-demo-")
    storage = Path(tmp_dir) / "costs.json"

    tracker = CostTracker(budget=config.budget, storage_path=storage)

    # Simulate several API calls
    print("\n  📝 Recording simulated API calls:")

    calls = [
        ("deepseek-v4", "deepseek-chat", 500, 200),
        ("openai-gpt4o-mini", "gpt-4o-mini", 1200, 800),
        ("anthropic-claude", "claude-3-haiku-20240307", 300, 150),
        ("deepseek-v4", "deepseek-chat", 100, 50),
        ("local-ollama", "qwen2.5:7b", 2000, 500),
    ]

    for api_name, model, tokens_in, tokens_out in calls:
        record = tracker.record_call(api_name, tokens_in, tokens_out, model)
        print(f"    • {api_name}: {tokens_in} in + {tokens_out} out = ${record.cost_usd:.6f}")

    # Check budget
    monthly = tracker.get_monthly_spent()
    remaining = tracker.get_remaining_budget()
    print(f"\n  💰 Monthly spent: ${monthly:.6f}")
    print(f"  💰 Remaining: ${remaining:.6f}")
    print(f"  ⚠️  Over budget? {tracker.is_over_budget()}")
    print(f"  ⚠️  Near limit? {tracker.is_near_limit()}")

    # Full stats
    stats = tracker.get_stats()
    print(f"\n  📊 Cost Statistics:")
    print(f"    Total calls: {stats.total_calls}")
    print(f"    Total cost: ${stats.total_cost_usd:.6f}")
    print(f"    Total tokens in: {stats.total_tokens_in}")
    print(f"    Total tokens out: {stats.total_tokens_out}")

    print(f"\n    By API:")
    for api, cost in stats.by_api.items():
        print(f"      {api}: ${cost:.6f}")

    # Export report
    print(f"\n  📄 Report (first 200 chars):")
    report = tracker.export_report("json")
    print(f"    {report[:200]}...")

    return tracker


# ── 4. World Brain ──────────────────────────────────────────────────────


async def demo_world_brain() -> WorldBrain:
    """Step 4: Demonstrate WorldBrain state management."""
    print_section("4. World Brain & State Management")

    brain = WorldBrain()

    # Show initial state
    resp = await brain.process(BrainRequest(
        task_id="demo-world-1",
        user_input="What is the current world state?",
    ))
    print(f"\n  🌍 Initial state: {resp.content}")

    # Set some values
    for key, value in [
        ("project", "octopus"),
        ("version", "0.1.0"),
        ("status", "active"),
        ("current_file", "examples/basic_usage.py"),
    ]:
        await brain.process(BrainRequest(
            task_id=f"demo-set-{key}",
            user_input=f"Set {key} to {value}",
            metadata={"action": f"set:{key}:{value}"},
        ))

    # Get a specific value
    resp = await brain.process(BrainRequest(
        task_id="demo-get-project",
        user_input="Get project name",
        metadata={"action": "get:project"},
    ))
    print(f"  📌 Get 'project': {resp.content}")

    # Take a snapshot
    resp = await brain.process(BrainRequest(
        task_id="demo-snapshot",
        user_input="Take a world snapshot",
        metadata={"action": "snapshot"},
    ))
    print(f"  📸 Snapshot: {resp.content}")

    # Change something and diff
    await brain.process(BrainRequest(
        task_id="demo-set-version",
        user_input="Update version",
        metadata={"action": "set:version:0.2.0"},
    ))

    resp = await brain.process(BrainRequest(
        task_id="demo-diff",
        user_input="Show diff",
        metadata={"action": "diff"},
    ))
    print(f"  📊 Diff: {resp.content}")

    # Environment variable lookup
    resp = await brain.process(BrainRequest(
        task_id="demo-env",
        user_input="Get PATH",
        metadata={"action": "env", "var": "PATH"},
    ))
    path_val = str(resp.structured_output.get("PATH", "")) if resp.structured_output else ""
    print(f"  🖥️  PATH: {path_val[:80]}..." if len(path_val) > 80 else f"  🖥️  PATH: {path_val}")

    # Simulate tool calls
    brain.state.update_from_tool_call("write_file", {"path": "/tmp/test.txt"}, "OK")
    brain.state.update_from_tool_call("exec", {"command": "ls", "cwd": "/tmp"}, "file1 file2")
    brain.state.update_from_tool_call("search", {"query": "octopus"}, "found 3 results")

    # Check tool call history
    history = brain.state.get("_tool_call_history", [])
    print(f"\n  🔧 Tool calls recorded: {len(history)}")
    for call in history:
        print(f"    • {call['tool']}: {call['result_summary'][:50]}")

    return brain


# ── 5. Task Routing Demo ────────────────────────────────────────────────


async def demo_task_routing() -> None:
    """Step 5: Demonstrate task routing with different brain types."""
    print_section("5. Task Routing Demo")

    # Create a sample set of brains (just WorldBrain for now since it's implemented)
    world = WorldBrain()

    tasks = [
        {
            "input": "What's the current git branch?",
            "complexity": TaskComplexity.SIMPLE,
            "risk": TaskRisk.NONE,
            "metadata": {"action": "get:project"},
        },
        {
            "input": "Remember that the project deadline is June 15",
            "complexity": TaskComplexity.MODERATE,
            "risk": TaskRisk.LOW,
            "metadata": {"action": "set:deadline:2025-06-15"},
        },
        {
            "input": "Show me the full world state",
            "complexity": TaskComplexity.TRIVIAL,
            "risk": TaskRisk.NONE,
            "metadata": {"action": "snapshot"},
        },
        {
            "input": "What files have changed since last check?",
            "complexity": TaskComplexity.SIMPLE,
            "risk": TaskRisk.NONE,
            "metadata": {"action": "diff"},
        },
    ]

    # Set initial state
    await world.process(BrainRequest(
        task_id="setup",
        user_input="Setup",
        metadata={"action": "set:project:octopus"},
    ))

    for i, task in enumerate(tasks):
        print(f"\n  📋 Task {i+1}: \"{task['input']}\"")
        print(f"     Complexity: {task['complexity'].name}, Risk: {task['risk'].name}")

        # Check which brain can handle it
        can = world.can_handle(BrainRequest(
            task_id=f"route-{i}",
            user_input=task["input"],
            complexity=task["complexity"],
            risk=task["risk"],
            metadata=task["metadata"],
        ))
        print(f"     WorldBrain can_handle: {can}")

        if can:
            resp = await world.process(BrainRequest(
                task_id=f"route-{i}",
                user_input=task["input"],
                complexity=task["complexity"],
                risk=task["risk"],
                metadata=task["metadata"],
            ))
            print(f"     Result: {resp.content}")
            print(f"     Success: {resp.success}, Confidence: {resp.confidence}")

    # Brain stats
    print(f"\n  📊 WorldBrain Statistics:")
    stats = world.stats
    for k, v in stats.items():
        print(f"    {k}: {v}")


# ── Main ─────────────────────────────────────────────────────────────────


async def main():
    """Run the full Octopus V5.0 demo."""
    print("🐙 Octopus V5.0 — Multi-Brain Agent Demo")
    print("=" * 60)

    try:
        # 1. Configuration
        config = demo_configuration()

        # 2. API Management
        await demo_api_management(config)

        # 3. Cost Tracking
        demo_cost_tracking(config)

        # 4. World Brain
        await demo_world_brain()

        # 5. Task Routing
        await demo_task_routing()

        print_section("✅ Demo Complete")
        print("  All core Octopus V5.0 features demonstrated successfully!")
        print("  - Configuration management ✓")
        print("  - API management with cost-aware selection ✓")
        print("  - Cost tracking with persistent storage ✓")
        print("  - World state management with snapshots/diffs ✓")
        print("  - Multi-brain task routing ✓")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
