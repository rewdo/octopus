"""Octopus CLI — the entry point for the multi-brain agent.

Usage:
    octopus init                    # Initialize workspace
    octopus run "summarize this"    # Run a task
    octopus status                  # Show agent status
    octopus config show             # Display config
    octopus skills list             # List skills
    octopus memory stats            # Memory statistics
    octopus memory search "query"   # Search memory
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from octopus.config import APIConfig, BudgetConfig, OctopusConfig

app = typer.Typer(
    name="octopus",
    help="Octopus — Multi-Brain Agent Infrastructure",
    add_completion=False,
)

console = Console()
DEFAULT_WORKSPACE = "./octopus-workspace"


# ── helpers ──────────────────────────────────────────────────────────────


def _get_config_path(workspace: str) -> Path:
    return Path(workspace) / "config.yaml"


def _load_config(workspace: str) -> OctopusConfig:
    """Load config or return default if not found."""
    path = _get_config_path(workspace)
    if path.exists():
        return OctopusConfig.from_file(path)
    return OctopusConfig.default()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ── init ─────────────────────────────────────────────────────────────────


@app.command()
def init(
    workspace: str = typer.Option(
        DEFAULT_WORKSPACE,
        "--workspace",
        "-w",
        help="Path to the Octopus workspace directory.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing config.",
    ),
):
    """Initialize an Octopus workspace with default configuration."""
    ws_path = Path(workspace)
    config_path = _get_config_path(workspace)

    if config_path.exists() and not force:
        console.print(f"[yellow]Workspace already exists at {ws_path.absolute()}[/yellow]")
        console.print("Use --force to overwrite.")
        raise typer.Exit(1)

    # Create workspace structure
    _ensure_dir(ws_path)
    _ensure_dir(ws_path / "skills")
    _ensure_dir(ws_path / "memory")
    _ensure_dir(ws_path / "logs")

    # Create default config with a demo API
    config = OctopusConfig.default()
    config.workspace_dir = ws_path.resolve()

    # Add sensible default APIs
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
    ]

    config.save(config_path)

    console.print(f"[green]✓[/green] Workspace initialized at {ws_path.absolute()}")
    console.print(f"  Config: {config_path}")
    console.print(f"  APIs configured: {len(config.apis)}")

    # Show configured APIs
    table = Table(title="Configured APIs")
    table.add_column("Name", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Price (1K in/out)", style="yellow")

    for api in config.apis:
        table.add_row(
            api.name,
            api.model,
            f"${api.price_per_1k_input:.6f} / ${api.price_per_1k_output:.6f}",
        )

    console.print(table)


# ── run ──────────────────────────────────────────────────────────────────


@app.command()
def run(
    task: str = typer.Argument(..., help="The task description to execute."),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Override the model to use.",
    ),
    workspace: str = typer.Option(
        DEFAULT_WORKSPACE,
        "--workspace",
        "-w",
        help="Path to the Octopus workspace directory.",
    ),
):
    """Run a task through the Octopus agent."""
    config = _load_config(workspace)

    if not config.apis:
        console.print("[red]No APIs configured. Run 'octopus init' first.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Task:[/bold cyan] {task}")
    if model:
        console.print(f"[dim]Model override: {model}[/dim]")

    # Determine which API to show (cheapest by default)
    from octopus.api import APIManager

    manager = APIManager(config)
    apis = manager.list_apis()

    console.print(f"\n[bold]Available APIs (sorted by cost):[/bold]")
    table = Table()
    table.add_column("Name", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Cost", style="yellow")
    table.add_column("Max Tokens", style="dim")

    for api in apis:
        cost = api.price_per_1k_input + api.price_per_1k_output
        table.add_row(api.name, api.model, f"${cost:.6f}/1K", str(api.max_tokens))

    console.print(table)
    console.print(
        f"\n[dim]Would route to cheapest API: [bold]{apis[0].name}[/bold] ({apis[0].model})[/dim]"
    )
    console.print(
        "[dim]Full agent execution (routing, planning, tool use) not yet wired in v0.1.[/dim]"
    )


# ── status ───────────────────────────────────────────────────────────────


@app.command()
def status(
    workspace: str = typer.Option(
        DEFAULT_WORKSPACE,
        "--workspace",
        "-w",
        help="Path to the Octopus workspace directory.",
    ),
):
    """Show agent status: active brains, memory stats, budget."""
    config = _load_config(workspace)

    # Brain status
    brain_table = Table(title="🧠 Brain Status")
    brain_table.add_column("Brain", style="cyan")
    brain_table.add_column("Backend", style="green")
    brain_table.add_column("Status", style="yellow")

    brains = [
        ("Cheap", config.brains.cheap),
        ("Skill", config.brains.skill),
        ("Planning", config.brains.planning),
        ("Frontier", config.brains.frontier),
        ("Memory", config.memory.graph_backend),
        ("World", "local_state"),
        ("Action", "local_exec"),
    ]

    for name, backend in brains:
        brain_table.add_row(name, backend, "✓ ready")

    console.print(brain_table)

    # Budget status
    budget_panel = Panel(
        f"Monthly budget: [bold]${config.budget.monthly_budget_usd:.2f}[/bold]\n"
        f"Max per task: [bold]${config.budget.max_per_task_usd:.2f}[/bold]\n"
        f"Warn at: [yellow]{config.budget.warn_threshold_pct:.0f}%[/yellow]\n"
        f"Tracking: {'[green]enabled[/green]' if config.budget.track_costs else '[red]disabled[/red]'}",
        title="💰 Budget",
    )
    console.print(budget_panel)

    # Memory stats
    memory_panel = Panel(
        f"Backend: [cyan]{config.memory.graph_backend}[/cyan]\n"
        f"Vector: [cyan]{config.memory.vector_backend}[/cyan] ({config.memory.vector_dimensions}d)\n"
        f"Working memory cap: {config.memory.working_memory_size} items\n"
        f"Importance threshold: {config.memory.importance_threshold}\n"
        f"GC interval: {config.memory.gc_interval_hours}h",
        title="🧠 Memory",
    )
    console.print(memory_panel)

    # Config info
    console.print(
        f"[dim]Config: {_get_config_path(workspace)} | "
        f"APIs: {len(config.apis)} | "
        f"Workspace: {config.workspace_dir}[/dim]"
    )


# ── config show ──────────────────────────────────────────────────────────


@app.command(name="config")
def config_show(
    workspace: str = typer.Option(
        DEFAULT_WORKSPACE,
        "--workspace",
        "-w",
        help="Path to the Octopus workspace directory.",
    ),
    format: str = typer.Option(
        "yaml",
        "--format",
        "-f",
        help="Output format: yaml or json.",
    ),
):
    """Display current configuration."""
    config = _load_config(workspace)

    if format == "json":
        data = config.model_dump()
        # Convert Path to string for JSON serialization
        data["workspace_dir"] = str(data["workspace_dir"])
        console.print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        import yaml

        data = config.model_dump()
        data["workspace_dir"] = str(data["workspace_dir"])
        console.print(yaml.dump(data, default_flow_style=False, allow_unicode=True))


# ── skills ───────────────────────────────────────────────────────────────

skills_app = typer.Typer(help="Skill management commands.")
app.add_typer(skills_app, name="skills")


@skills_app.command("list")
def skills_list(
    category: Optional[str] = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter skills by category.",
    ),
    workspace: str = typer.Option(
        DEFAULT_WORKSPACE,
        "--workspace",
        "-w",
    ),
):
    """List available skills."""
    skills_dir = Path(workspace) / "skills"
    if not skills_dir.exists():
        console.print("[yellow]No skills directory found. Run 'octopus init' first.[/yellow]")
        return

    # List skill directories
    skill_dirs = [d for d in skills_dir.iterdir() if d.is_dir()]
    if not skill_dirs:
        console.print("[dim]No skills installed yet.[/dim]")
        console.print(f"Add skills to: {skills_dir}")
        return

    table = Table(title="📦 Available Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="dim")

    for sd in sorted(skill_dirs):
        table.add_row(sd.name, str(sd))

    console.print(table)
    console.print(f"\n[dim]{len(skill_dirs)} skill(s) total[/dim]")


# ── memory ───────────────────────────────────────────────────────────────

memory_app = typer.Typer(help="Memory management commands.")
app.add_typer(memory_app, name="memory")


@memory_app.command("stats")
def memory_stats(
    workspace: str = typer.Option(DEFAULT_WORKSPACE, "--workspace", "-w"),
):
    """Show memory statistics."""
    config = _load_config(workspace)
    memory_dir = Path(workspace) / "memory"

    console.print("[bold]Memory System Stats[/bold]")
    console.print(f"  Backend: [cyan]{config.memory.graph_backend}[/cyan]")
    console.print(f"  Vector: [cyan]{config.memory.vector_backend}[/cyan]")
    console.print(f"  Dimensions: {config.memory.vector_dimensions}")
    console.print(f"  Working memory cap: {config.memory.working_memory_size}")
    console.print(f"  Importance threshold: {config.memory.importance_threshold}")
    console.print(f"  GC interval: {config.memory.gc_interval_hours}h")

    if memory_dir.exists():
        files = list(memory_dir.glob("**/*"))
        console.print(f"\n  Memory files on disk: [bold]{len(files)}[/bold]")
        for f in sorted(files)[:10]:
            size = f.stat().st_size if f.is_file() else 0
            console.print(f"    {f.name} ({size}B)" if f.is_file() else f"    {f.name}/")
        if len(files) > 10:
            console.print(f"    ... and {len(files) - 10} more")
    else:
        console.print("  [dim]No memory directory yet. Run 'octopus init'.[/dim]")


@memory_app.command("search")
def memory_search_cmd(
    query: str = typer.Argument(..., help="Search query for the memory graph."),
    workspace: str = typer.Option(DEFAULT_WORKSPACE, "--workspace", "-w"),
):
    """Search the memory graph."""
    console.print(f"[bold]Searching memory for:[/bold] {query}")
    console.print("[dim]Memory search backend not yet wired in v0.1.[/dim]")
    console.print("[dim]Query would be vectorized and matched against stored memories.[/dim]")


# ── main ─────────────────────────────────────────────────────────────────


def main():
    """Entry point for the Octopus CLI."""
    app()


if __name__ == "__main__":
    main()
