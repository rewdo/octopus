# 更新日志

> 📖 [English version](CHANGELOG.md)

Octopus 项目的所有重要变更都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)，
本项目遵循 [语义化版本](https://semver.org/spec/v2.0.0.html)。

## [Unreleased]（未发布）

### Added（新增）
- 即将推出的功能占位

## [0.1.0] — 2026-05-26

### Added（新增）
- 项目骨架和包结构（`src/octopus/`）
- 基于 Pydantic 模型的配置系统（`OctopusConfig`）
  - API 端点配置，支持按 token 计费
  - Token 预算与成本控制（`BudgetConfig`）
  - 大脑执行后端配置（`BrainConfig`）
  - 记忆系统配置（`MemoryConfig`）
  - 路由器阈值与权重配置
  - YAML/JSON 配置文件加载与保存
- 基础大脑接口（`BaseBrain`），含标准化的 `BrainRequest` / `BrainResponse` 协议
- 七种大脑类型定义：廉价脑、技能脑、记忆脑、规划脑、行动脑、世界脑、前沿脑
- 任务复杂度与风险分类枚举
- 带 SIMD 加速的 CPU 优化向量数学模块
- 包元数据与构建配置（`pyproject.toml`）
- 开发工具链：ruff、mypy、pytest 配置
- CI/CD 流水线（GitHub Actions）
- 文档：README、CONTRIBUTING、CODE_OF_CONDUCT、SECURITY、LICENSE

[Unreleased]: https://github.com/octopus-agent/octopus/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/octopus-agent/octopus/releases/tag/v0.1.0
