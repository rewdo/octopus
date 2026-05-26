# 参与贡献 Octopus

> 📖 [English version](CONTRIBUTING.md)

感谢你对贡献 Octopus 的兴趣！Octopus 是一个社区驱动的项目，我们欢迎各种形式的贡献 — 代码、文档、Bug 报告、功能想法等等。

## 行为准则

本项目遵循 [贡献者公约行为准则](CODE_OF_CONDUCT.md)。请在参与前阅读。

## 如何贡献

### 1. 找到可以做的事

- 查看 [Issues](https://github.com/octopus-agent/octopus/issues) 中待解决的任务
- 寻找标有 `good first issue` 或 `help wanted` 标签的 issue
- 通过提交 [功能请求](https://github.com/octopus-agent/octopus/issues/new?template=feature_request.md) 来提议新特性

### 2. 开发工作流

```bash
# 1. Fork 仓库（通过 GitHub 网页界面操作）

# 2. 克隆你的 fork
git clone https://github.com/YOUR_USERNAME/octopus.git
cd octopus

# 3. 创建分支
git checkout -b feat/my-feature
# 分支命名规范：feat/、fix/、docs/、refactor/、test/、chore/

# 4. 搭建开发环境
pip install -e ".[dev]"

# 5. 做出你的改动
# 6. 运行检查
ruff check .
mypy src/
pytest

# 7. 提交并推送
git add .
git commit -m "feat: 添加我的功能"
git push origin feat/my-feature

# 8. 在 GitHub 上发起 Pull Request
```

### 3. 提交约定

我们遵循 [约定式提交（Conventional Commits）](https://www.conventionalcommits.org/)：

| 前缀 | 用途 |
|--------|-------|
| `feat:` | 新功能 |
| `fix:` | Bug 修复 |
| `docs:` | 文档变更 |
| `refactor:` | 代码重构（不改变行为） |
| `test:` | 添加或更新测试 |
| `chore:` | 构建、CI、依赖更新 |
| `style:` | 格式化、缺失分号等 |

示例：
```
feat: 添加对 Ollama 本地模型的支持
fix: 修复向量存储中的内存泄漏
docs: 更新 README 添加配置示例
refactor: 将 token 计数提取到工具模块
test: 添加 CognitiveRouter 的集成测试
chore: 将 pydantic 升级到 v2.5
```

### 4. 代码风格

- **Linter：** [Ruff](https://docs.astral.sh/ruff/) — 行长度 100，目标 Python 3.10
- **类型检查：** [Mypy](https://mypy-lang.org/) — 新代码使用严格模式
- **格式化工具：** Ruff 内建格式化器（兼容 Black）
- **文档字符串：** 推荐 Google 风格

提交前运行：

```bash
ruff check . && ruff format --check . && mypy src/ && pytest
```

### 5. 测试

- 为所有新功能编写测试
- 将测试放在 `tests/` 目录中，镜像 `src/octopus/` 的结构
- 使用 `pytest` 和 `pytest-asyncio` 进行异步测试
- 新代码覆盖率目标 >80%

```bash
# 运行所有测试
pytest

# 带覆盖率运行
pytest --cov=octopus --cov-report=term-missing
```

### 6. Pull Request 指南

- **保持 PR 聚焦。** 每个 PR 只包含一个功能或修复。
- **写清晰的描述。** 解决了什么问题？如何解决的？
- **关联相关 issue。** 使用 `Closes #123` 或 `Refs #123`。
- **确保 CI 通过。** 所有检查必须在 review 前通过。
- **更新 CHANGELOG.md**，如果你的变更是面向用户的，请在 `[Unreleased]` 下添加条目。
- **保持耐心。** 审核者都是志愿者。等待几天后可以适当提醒。

## 项目结构

```
octopus/
├── src/octopus/
│   ├── __init__.py       # 包入口点
│   ├── config.py         # 配置系统（Pydantic）
│   ├── router/           # 认知路由器（任务路由）
│   ├── brains/           # 7 个专业大脑
│   ├── memory/           # 图谱 + 向量记忆存储
│   ├── skills/           # 技能引擎和技能包
│   ├── world/            # 世界状态管理
│   └── api/              # API 和 CLI 接口
├── tests/                # 测试套件
├── docs/                 # 扩展文档
├── examples/             # 使用示例
├── pyproject.toml        # 项目元数据和工具配置
└── README.md             # 你在看的就是这个
```

## 获取帮助

- **有问题？** 发起 [Discussion](https://github.com/octopus-agent/octopus/discussions)
- **发现 Bug？** 提交 [Issue](https://github.com/octopus-agent/octopus/issues/new?template=bug_report.md)
- **想聊天？** 加入我们的社区（链接即将上线）

---

感谢你为 Octopus 做出贡献！🐙
