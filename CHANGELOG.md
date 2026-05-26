# Changelog

All notable changes to the Octopus project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Placeholder for upcoming features

## [0.1.0] — 2026-05-26

### Added
- Project skeleton and package structure (`src/octopus/`)
- Configuration system with Pydantic models (`OctopusConfig`)
  - API endpoint configuration with per-token pricing
  - Token budget and cost control (`BudgetConfig`)
  - Brain execution backend configuration (`BrainConfig`)
  - Memory system configuration (`MemoryConfig`)
  - Router threshold and weight configuration
  - YAML/JSON config file loading and saving
- Base brain interface (`BaseBrain`) with standardized `BrainRequest` / `BrainResponse` protocol
- Seven brain type definitions: Cheap, Skill, Memory, Planning, Action, World, Frontier
- Task complexity and risk classification enums
- CPU-optimized vector math module with SIMD acceleration
- Package metadata and build configuration (`pyproject.toml`)
- Development tooling: ruff, mypy, pytest configuration
- CI/CD pipeline (GitHub Actions)
- Documentation: README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, LICENSE

[Unreleased]: https://github.com/octopus-agent/octopus/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/octopus-agent/octopus/releases/tag/v0.1.0
