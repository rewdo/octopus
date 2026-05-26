# Octopus v0.1.0 — First Release

🐙 **Octopus v0.1.0** — Multi-Brain Agent Infrastructure Phase 1 Core OS

## Highlights

- **7 specialized brains**: Cheap, Skill, Action, Memory, Planning, Frontier, World
- **Cognitive Router**: 9-dimension scoring engine with T1/T2/T3 threshold routing
- **4-layer memory**: Working, Episodic, Semantic, Procedural — with NetworkX graph storage
- **Context Compiler**: On-demand minimal context assembly, preventing context pollution
- **30+ pre-built skills**: 6 categories (text, data, code, file, web, utility)
- **Multi-tier API**: Configurable API endpoints with per-token pricing and cost tracking
- **Self-healing**: Automatic retry with exponential backoff, checkpoint/resume
- **Verification layer**: Code syntax checking, fact contradiction detection
- **Skill distillation**: Pattern extraction from task traces → reusable skill candidates
- **Cognitive budget**: Expected-gain formula for cost-aware brain selection
- **Kimi API integration**: Real LLM calls through Frontier Brain (moonshot-v1 models)

## What's Included

| Module | Description |
|--------|-------------|
| `octopus.router` | Cognitive Router with 9-dim scoring |
| `octopus.brains` | 7 brain implementations |
| `octopus.memory` | 4-layer memory + graph + context compiler |
| `octopus.skills` | Skill engine + 30+ skill bundles + distiller |
| `octopus.api` | API manager + cost tracker |
| `octopus.budget` | Cognitive budget system |
| `octopus.heal` | Self-healing + retry + checkpoint |
| `octopus.verify` | Code + fact verification |
| `octopus.world` | World state engine |
| `octopus.cli` | 6 CLI commands (init/run/status/config/skills/memory) |

## Quick Start

```bash
git clone https://github.com/rewdo/octopus.git
cd octopus
pip install -e .
export KIMI_API_KEY="your-key"
octopus init
octopus run "hello"
```

## Git Tags (Phase Rollback)

| Tag | Description | Checkout |
|-----|-------------|----------|
| `phase-1` | Core OS | `git checkout phase-1` |
| `phase-2` | Self-Growing | `git checkout phase-2` |
| `v0.1.0` | Full Octopus (latest) | `git checkout v0.1.0` |

- Skill benchmark arena
- LLM-powered Planning Brain decomposition
- Docker image + PyPI package
- Personal ontology construction
- Multi-user isolation
