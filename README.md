# 🐙 Octopus — Multi-Brain Agent Infrastructure

> 📖 [中文版](README.zh-CN.md)

<p align="center">
  <strong>Token-Economic · Never-Forgetting · Self-Evolving</strong><br>
  <em>A cognitive operating system for AI agents — like an octopus, with multiple brains.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="Status">
  <a href="https://github.com/octopus-agent/octopus/actions"><img src="https://img.shields.io/badge/CI-passing-brightgreen" alt="CI"></a>
</p>

---

## What is Octopus?

Octopus is a **multi-brain agent infrastructure** that treats every cloud LLM call as a scarce currency. Instead of dumping everything into a single monolithic prompt, Octopus routes each task through a network of specialized cognitive modules — **7 brains**, each optimized for a different class of work.

**Core beliefs:**

- **Token is currency.** Cloud LLMs are expensive. Octopus exhausts every local option before spending a single API token.
- **Skill > Prompt.** Reusable, testable skills outperform ad-hoc prompts every time.
- **Memory ≠ Context.** Memory lives in a graph+vector store. Context is compiled on demand — lean, relevant, and cheap.
- **Verify everything.** No single LLM output is trusted without validation. Cross-checking is built into the architecture.

## Architecture

```
                          ┌─────────────────────┐
                          │   Cognitive Router   │
                          │  (9-dimension score)  │
                          └──────────┬──────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │              │           │           │              │
    ┌─────▼─────┐  ┌─────▼─────┐  ┌──▼───┐  ┌───▼────┐  ┌─────▼─────┐
    │   Cheap   │  │   Skill   │  │Memory│  │ World  │  │  Action   │
    │   Brain   │  │   Brain   │  │ Brain│  │ Brain  │  │   Brain   │
    │  (local)  │  │  (local)  │  │(RAG) │  │(state) │  │  (tools)  │
    └───────────┘  └───────────┘  └──────┘  └────────┘  └───────────┘
          │              │           │           │              │
          └──────────────┴───────────┴───────────┴──────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   Planning Brain     │
                          │  (decomposition)     │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   Frontier Brain     │
                          │   (cloud LLM)        │
                          │   LAST RESORT        │
                          └─────────────────────┘
```

**Routing Decision Flow:**

```
User Input
    │
    ▼
Context Compiler ──► Memory Graph ──► Relevant Memories
    │
    ▼
9-Dimension Scoring:
  S = α·Complexity + β·Novelty + γ·Risk + δ·Realtime
    - ε·SkillConfidence - ζ·LocalCapability - η·BudgetRemaining
    │
    ├── S < T1  ──► Cheap / Skill Brain (local, zero-cost)
    ├── T1 ≤ S < T2 ──► Planning + Skill + Local
    ├── T2 ≤ S < T3 ──► Hybrid (local + compressed cloud)
    └── S ≥ T3  ──► Frontier Brain (full cloud LLM)
```

## The 7 Brains

| # | Brain | Role | Execution Backend | Cost |
|---|-------|------|-------------------|------|
| 1 | **Cheap Brain** | Rule-based patterns, regex, tiny local models | Local (on-device) | **Free** |
| 2 | **Skill Brain** | Pre-compiled, testable skill execution | Local engine | **Free** |
| 3 | **Memory Brain** | Long-term memory retrieval (graph + vector) | RAG (local/remote) | Low |
| 4 | **Planning Brain** | Task decomposition, multi-step orchestration | Mid-tier API | Medium |
| 5 | **Action Brain** | Tool execution (shell, browser, API) | Local sandbox | Low |
| 6 | **World Brain** | World state maintenance, persistent tracking | Local state | **Free** |
| 7 | **Frontier Brain** | High-value reasoning, creative synthesis | Cloud LLM | High |

## Quick Start

### Installation

```bash
pip install octopus-agent
```

Or from source:

```bash
git clone https://github.com/octopus-agent/octopus.git
cd octopus
pip install -e ".[dev]"
```

### Minimal Example

```python
from octopus import OctopusConfig, CognitiveRouter

# Load config (YAML, JSON, or defaults)
config = OctopusConfig.default()
config.workspace_dir = "./my-octopus-workspace"

# Create router
router = CognitiveRouter(config)

# Process a task — routing is automatic
result = await router.process("Summarize recent conversations about the Q4 budget")
print(result.content)
print(f"Tokens used: {result.tokens_used}, Cost: ${result.cost_usd:.4f}")
```

### Configuration

```yaml
# octopus.yaml
config_version: "1.0"
workspace_dir: ./octopus-workspace

apis:
  - name: "DeepSeek V4"
    provider: deepseek
    base_url: https://api.deepseek.com/v1
    api_key: $DEEPSEEK_API_KEY
    model: deepseek-chat
    price_per_1k_input: 0.14
    price_per_1k_output: 0.28
    priority: 1

budget:
  monthly_budget_usd: 10.0
  max_per_task_usd: 0.10

brains:
  cheap: local_rule
  skill: local_engine
  planning: api_mid
  frontier: api_high

memory:
  graph_backend: networkx
  vector_backend: chromadb
  working_memory_size: 50
```

Load it:

```python
config = OctopusConfig.from_file("octopus.yaml")
```

## Key Concepts

### Token Economy

Octopus implements a **progressive escalation** strategy:

```
Rule Match (0 tokens)
  → Pre-compiled Skill (0 tokens)
    → Local Small Model (~0 tokens, on-device)
      → Vector RAG (few tokens, compression)
        → Compressed Cloud Call (moderate tokens)
          → Full Cloud LLM (last resort)
```

Every decision is cost-aware. The router weighs token cost against expected value before escalating.

### Memory Graph (Never-Forgetting)

Memory is a dual-store system:

- **Graph store** (NetworkX / Neo4j): Entities, relationships, temporal links
- **Vector store** (ChromaDB): Semantic embeddings for similarity search

The **Context Compiler** assembles context on-demand from the memory graph — only what's relevant, never raw history dumps.

### Skill Engine (Skill > Prompt)

Skills are reusable, versioned, testable modules:

```
skills/
  ├── summarization/
  │   ├── skill.yaml      # Metadata, version, test cases
  │   └── prompt.txt      # Optimized template
  ├── code_review/
  └── translation/
```

Skills have confidence scores that feed back into the router's scoring function.

### Multi-Agent Verification

Critical outputs from the Frontier Brain can be cross-validated by a second model or by rule-based checks. No single LLM is blindly trusted.

## Project Status & Roadmap

| Phase | Milestone | Status |
|-------|-----------|--------|
| **Phase 1** | Core skeleton: config, brain base, router stub, memory store | 🚧 In Progress |
| **Phase 2** | Cheap Brain + Skill Brain (local execution) | ⬜ Planned |
| **Phase 3** | Memory Brain (graph + vector RAG) | ⬜ Planned |
| **Phase 4** | Planning Brain + Action Brain (tool use) | ⬜ Planned |
| **Phase 5** | World Brain + Multi-Agent Verification | ⬜ Planned |
| **Phase 6** | CLI + Web Dashboard + Skill Marketplace | ⬜ Planned |
| **Phase 7** | Self-evolution (auto-skill generation, meta-learning) | ⬜ Planned |

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Quick start for developers:

```bash
git clone https://github.com/octopus-agent/octopus.git
cd octopus
pip install -e ".[dev]"
ruff check . && mypy src/ && pytest
```

## License

MIT © 2026 Octopus Contributors. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with 🐙 by the Octopus community</sub>
</p>
