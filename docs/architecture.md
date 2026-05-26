# Octopus Architecture

> Multi-Brain Agent Infrastructure — v0.1.0

## Overview

Octopus is a token-economic, never-forgetting, self-evolving cognitive OS for AI agents. Like an octopus with multiple brains, different tasks are handled by specialized cognitive modules coordinated by a central router.

```
User Input → Cognitive Router → Brain Selection → Execution → Response
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
               Cheap Brain   Skill Brain   Action Brain  ... (7 total)
```

## The Seven Brains

| Brain | Type | Role | Typical Tasks |
|-------|------|------|---------------|
| Cheap | Rule-based | Intent classification, entity extraction | Greetings, simple lookups |
| Skill | Workflow engine | Execute pre-compiled skill DAGs | Text summarization, data extraction |
| Action | Tool executor | Shell, file, web operations | File listing, web search |
| Memory | Graph search | Long-term memory retrieval | "What did I fix last week?" |
| Planning | Task decomposition | Break complex tasks into sub-task DAGs | "Build and deploy a web app" |
| Frontier | LLM gateway | High-capability cloud reasoning | Creative writing, complex analysis |
| World | State tracker | Environment snapshot queries | "Show current state", "What OS?" |

## Cognitive Router

The router scores each task across 9 dimensions:

```
FinalScore = α·Complexity + β·Novelty + γ·Risk + δ·RealtimeNeed
             - ε·SkillConfidence - ζ·LocalCapability - η·BudgetRemaining
```

| Threshold | Range | Target Brain |
|-----------|-------|--------------|
| T1 (2.0) | Below T1 | Cheap / Skill |
| T2 (5.0) | T1–T2 | Planning + Skill |
| T3 (7.0) | T2–T3 | Hybrid (Planning + Frontier) |
| — | Above T3 | Frontier |

## Memory System

Four-layer hierarchical memory:

| Layer | Name | Storage | Lifetime |
|-------|------|---------|----------|
| L1 | Working | In-memory dict | Minutes |
| L2 | Episodic | NetworkX graph + timeline | Long-term |
| L3 | Semantic | Vector + property graph | Long-term |
| L4 | Procedural | Skill bundles | Permanent |

**Memory ≠ Context** — stored memory never enters the LLM context automatically. The Context Compiler builds the minimal necessary context on demand.

## Skill System

Skills are executable DAGs, not prompts:

```
Draft → Evaluate (cost/success) → Publish → Monitor → Optimize → Archive
```

30+ pre-built skills across 6 categories: text, data, code, file, web, utility.

## Data Flow

```
User Input
  │
  ▼
Router (9-dim scoring)
  │
  ├─ Context Compiler → Memory Graph
  │
  ▼
Brain Request (compiled context)
  │
  ▼
Brain Execution
  │
  ├─ Verification Layer
  ├─ Cost Tracker
  └─ Episodic Memory Update
  │
  ▼
Response
```

## Configuration

Multi-tier API with per-token pricing:

```yaml
apis:
  - name: kimi-cheap
    base_url: https://api.moonshot.cn/v1
    api_key: $KIMI_API_KEY
    model: moonshot-v1-8k
    price_per_1k_input: 0.0017
```

## Project Structure

```
octopus/
├── src/octopus/
│   ├── router/          # Cognitive Router
│   ├── brains/          # 7 brain implementations
│   ├── memory/          # Memory graph + layers + compiler
│   ├── skills/          # Skill engine + bundles
│   ├── api/             # API manager + cost tracker
│   ├── budget/          # Cognitive budget system
│   ├── heal/            # Self-healing + retry
│   ├── verify/          # Verification layer
│   └── world/           # World state engine
├── tests/               # Test suite
├── docs/                # Documentation
└── examples/            # Usage examples
```
