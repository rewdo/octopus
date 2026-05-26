# Contributing to Octopus

Thank you for your interest in contributing! Octopus is a community-driven project, and we welcome contributions of all kinds — code, documentation, bug reports, feature ideas, and more.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating.

## How to Contribute

### 1. Find Something to Work On

- Check the [Issues](https://github.com/octopus-agent/octopus/issues) for open tasks
- Look for issues labeled `good first issue` or `help wanted`
- Propose a new feature by opening a [Feature Request](https://github.com/octopus-agent/octopus/issues/new?template=feature_request.md)

### 2. Development Workflow

```bash
# 1. Fork the repository (via GitHub UI)

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/octopus.git
cd octopus

# 3. Create a branch
git checkout -b feat/my-feature
# Branch naming: feat/, fix/, docs/, refactor/, test/, chore/

# 4. Set up development environment
pip install -e ".[dev]"

# 5. Make your changes
# 6. Run checks
ruff check .
mypy src/
pytest

# 7. Commit and push
git add .
git commit -m "feat: add my feature"
git push origin feat/my-feature

# 8. Open a Pull Request on GitHub
```

### 3. Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Usage |
|--------|-------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation changes |
| `refactor:` | Code restructuring (no behavior change) |
| `test:` | Adding or updating tests |
| `chore:` | Build, CI, dependency updates |
| `style:` | Formatting, missing semicolons, etc. |

Examples:
```
feat: add support for Ollama local models
fix: resolve memory leak in vector store
docs: update README with configuration examples
refactor: extract token counting to utility module
test: add integration tests for CognitiveRouter
chore: bump pydantic to v2.5
```

### 4. Code Style

- **Linter:** [Ruff](https://docs.astral.sh/ruff/) — line length 100, target Python 3.10
- **Type checking:** [Mypy](https://mypy-lang.org/) — strict mode for new code
- **Formatter:** Ruff's built-in formatter (compatible with Black)
- **Docstrings:** Google style preferred

Run before committing:

```bash
ruff check . && ruff format --check . && mypy src/ && pytest
```

### 5. Testing

- Write tests for all new functionality
- Place tests in the `tests/` directory, mirroring the `src/octopus/` structure
- Use `pytest` with `pytest-asyncio` for async tests
- Aim for >80% coverage on new code

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=octopus --cov-report=term-missing
```

### 6. Pull Request Guidelines

- **Keep PRs focused.** One feature or fix per PR.
- **Write a clear description.** What problem does it solve? How?
- **Link related issues.** Use `Closes #123` or `Refs #123`.
- **Ensure CI passes.** All checks must be green before review.
- **Update CHANGELOG.md** under `[Unreleased]` if your change is user-facing.
- **Be patient.** Reviewers are volunteers. Feel free to ping after a few days.

## Project Structure

```
octopus/
├── src/octopus/
│   ├── __init__.py       # Package entry point
│   ├── config.py         # Configuration system (Pydantic)
│   ├── router/           # Cognitive Router (task routing)
│   ├── brains/           # 7 specialized brains
│   ├── memory/           # Graph + vector memory store
│   ├── skills/           # Skill engine and bundles
│   ├── world/            # World state management
│   └── api/              # API and CLI interfaces
├── tests/                # Test suite
├── docs/                 # Extended documentation
├── examples/             # Usage examples
├── pyproject.toml        # Project metadata and tool config
└── README.md             # You are here
```

## Getting Help

- **Questions?** Open a [Discussion](https://github.com/octopus-agent/octopus/discussions)
- **Bug?** Open an [Issue](https://github.com/octopus-agent/octopus/issues/new?template=bug_report.md)
- **Chat?** Join our community (link coming soon)

---

Thank you for contributing to Octopus! 🐙
