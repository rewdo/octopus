# 🐙 Octopus — 多脑智能体基础设施

> 📖 [English version](README.md)

<p align="center">
  <strong>Token 即货币 · 永不遗忘 · 自我进化</strong><br>
  <em>面向 AI 智能体的认知操作系统 — 如章鱼般，拥有多个大脑。</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="Status">
  <a href="https://github.com/octopus-agent/octopus/actions"><img src="https://img.shields.io/badge/CI-passing-brightgreen" alt="CI"></a>
</p>

---

## 什么是 Octopus？

Octopus 是一个**多脑智能体基础设施**，将每一次云端 LLM 调用视为稀缺货币。它不会把所有内容一股脑塞进单个巨型 prompt，而是将每个任务路由到一个由专业认知模块组成的网络 — **7 个大脑**，每个大脑针对不同类型的工作进行了优化。

**核心理念：**

- **Token 即货币。** 云端 LLM 开销不菲。Octopus 在花费哪怕一个 API token 之前，会穷尽所有本地选项。
- **技能优于 Prompt。** 可复用、可测试的技能，每次都胜过临时拼凑的 prompt。
- **记忆 ≠ 上下文。** 记忆存储在知识图谱 + 向量存储中。上下文按需编译 — 精简、相关、便宜。
- **一切皆需验证。** 任何单一 LLM 的输出都不被盲目信任。交叉校验内建于架构之中。

## 架构

```
                          ┌─────────────────────┐
                          │    认知路由器        │
                          │  （9 维度评分）       │
                          └──────────┬──────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │              │           │           │              │
    ┌─────▼─────┐  ┌─────▼─────┐  ┌──▼───┐  ┌───▼────┐  ┌─────▼─────┐
    │   廉价脑   │  │   技能脑   │  │ 记忆脑│  │  世界脑 │  │   行动脑   │
    │  （本地）  │  │  （本地）  │  │（RAG）│  │（状态） │  │  （工具）  │
    └───────────┘  └───────────┘  └──────┘  └────────┘  └───────────┘
          │              │           │           │              │
          └──────────────┴───────────┴───────────┴──────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │      规划脑          │
                          │   （任务分解）        │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │      前沿脑          │
                          │   （云端 LLM）       │
                          │     最后手段          │
                          └─────────────────────┘
```

**路由决策流程：**

```
用户输入
    │
    ▼
上下文编译器 ──► 记忆图 ──► 相关记忆
    │
    ▼
9 维度评分：
  S = α·复杂度 + β·新颖度 + γ·风险 + δ·实时性
    - ε·技能置信度 - ζ·本地能力 - η·预算剩余
    │
    ├── S < T1  ──► 廉价脑 / 技能脑（本地，零成本）
    ├── T1 ≤ S < T2 ──► 规划脑 + 技能脑 + 本地
    ├── T2 ≤ S < T3 ──► 混合模式（本地 + 压缩云端）
    └── S ≥ T3  ──► 前沿脑（完整云端 LLM）
```

## 7 个大脑

| # | 大脑 | 职责 | 执行后端 | 成本 |
|---|------|------|-------------------|------|
| 1 | **廉价脑** | 基于规则的模式匹配、正则、微型本地模型 | 本地（设备端） | **免费** |
| 2 | **技能脑** | 预编译、可测试的技能执行 | 本地引擎 | **免费** |
| 3 | **记忆脑** | 长期记忆检索（图谱 + 向量） | RAG（本地/远程） | 低 |
| 4 | **规划脑** | 任务分解、多步骤编排 | 中端 API | 中 |
| 5 | **行动脑** | 工具执行（Shell、浏览器、API） | 本地沙盒 | 低 |
| 6 | **世界脑** | 世界状态维护、持久化追踪 | 本地状态 | **免费** |
| 7 | **前沿脑** | 高价值推理、创造性综合 | 云端 LLM | 高 |

## 快速开始

### 安装

```bash
pip install octopus-agent
```

或从源码安装：

```bash
git clone https://github.com/octopus-agent/octopus.git
cd octopus
pip install -e ".[dev]"
```

### 最小示例

```python
from octopus import OctopusConfig, CognitiveRouter

# 加载配置（YAML、JSON 或默认值）
config = OctopusConfig.default()
config.workspace_dir = "./my-octopus-workspace"

# 创建路由器
router = CognitiveRouter(config)

# 处理任务 — 路由自动完成
result = await router.process("总结关于 Q4 预算的近期对话")
print(result.content)
print(f"Token 用量：{result.tokens_used}，费用：${result.cost_usd:.4f}")
```

### 配置

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

加载配置：

```python
config = OctopusConfig.from_file("octopus.yaml")
```

## 核心概念

### Token 经济

Octopus 采用**渐进式升级**策略：

```
规则匹配（0 token）
  → 预编译技能（0 token）
    → 本地小模型（~0 token，设备端）
      → 向量 RAG（少量 token，压缩）
        → 压缩云端调用（适度 token）
          → 完整云端 LLM（最后手段）
```

每个决策都带有成本意识。路由器在升级前会权衡 token 成本与预期价值。

### 记忆图（永不遗忘）

记忆采用双存储系统：

- **图谱存储**（NetworkX / Neo4j）：实体、关系、时序链接
- **向量存储**（ChromaDB）：用于相似度搜索的语义嵌入

**上下文编译器**按需从记忆图中组装上下文 — 只取相关内容，绝不生搬原始历史记录。

### 技能引擎（技能优于 Prompt）

技能是可复用、有版本管理、可测试的模块：

```
skills/
  ├── summarization/
  │   ├── skill.yaml      # 元数据、版本、测试用例
  │   └── prompt.txt      # 优化后的模板
  ├── code_review/
  └── translation/
```

技能具有置信度评分，该评分会反馈到路由器的评分函数中。

### 多智能体验证

前沿脑的关键输出可由第二个模型或基于规则的检查进行交叉验证。没有任何单一 LLM 的输出会被盲目信任。

## 项目状态与路线图

| 阶段 | 里程碑 | 状态 |
|-------|-----------|--------|
| **Phase 1** | 核心骨架：配置、大脑基类、路由器桩、记忆存储 | 🚧 进行中 |
| **Phase 2** | 廉价脑 + 技能脑（本地执行） | ⬜ 计划中 |
| **Phase 3** | 记忆脑（图谱 + 向量 RAG） | ⬜ 计划中 |
| **Phase 4** | 规划脑 + 行动脑（工具使用） | ⬜ 计划中 |
| **Phase 5** | 世界脑 + 多智能体验证 | ⬜ 计划中 |
| **Phase 6** | CLI + Web 仪表盘 + 技能市场 | ⬜ 计划中 |
| **Phase 7** | 自我进化（自动技能生成、元学习） | ⬜ 计划中 |

## 参与贡献

欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

开发者快速开始：

```bash
git clone https://github.com/octopus-agent/octopus.git
cd octopus
pip install -e ".[dev]"
ruff check . && mypy src/ && pytest
```

## 许可证

MIT © 2026 Octopus Contributors。详见 [LICENSE](LICENSE)。

---

<p align="center">
  <sub>由 Octopus 社区以 🐙 构建</sub>
</p>
