"""
Planning Brain — Complex task decomposition into subtask DAGs.

Phase 1: Rule-based template matching for common patterns.
Phase 2 (future): LLM-powered decomposition via API Manager.

Responsibilities:
    - Analyze tasks for multi-step structure
    - Decompose into SubTask DAG with dependencies
    - Topological sort for execution order
    - Assign suggested brains per subtask
    - Return structured Plan as BrainResponse.structured_output
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from .base import (
    BaseBrain,
    BrainRequest,
    BrainResponse,
    BrainType,
    TaskComplexity,
)


# ── Data types ──────────────────────────────────────────────────────────────


@dataclass
class SubTask:
    """A single unit of work within a decomposed plan.

    Each SubTask represents one atomic step that can be dispatched to
    a specific brain for execution. Dependencies form a DAG that
    determines execution order.
    """

    id: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    estimated_complexity: str = "simple"  # trivial|simple|moderate|complex
    suggested_brain: str = "cheap"  # cheap|skill|action|memory|planning|frontier
    input_hint: str = ""  # What this subtask expects as input
    output_hint: str = ""  # What this subtask produces


@dataclass
class Plan:
    """Structured decomposition result from PlanningBrain.

    Contains the full set of subtasks, their dependency DAG, and a
    topologically sorted execution order. This is returned as
    BrainResponse.structured_output.
    """

    task_id: str
    original_input: str
    subtasks: list[SubTask]
    execution_order: list[str]  # Topologically sorted subtask IDs
    estimated_total_complexity: str = "moderate"
    plan_type: str = "rule"  # "rule" | "llm"
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Decomposition templates (Phase 1: rule-based) ───────────────────────────


@dataclass
class DecompositionTemplate:
    """A rule-based template for decomposing a known task pattern."""

    name: str
    patterns: list[str]  # Regex patterns to match user input
    keywords: list[str]  # Simpler keyword triggers
    subtask_factory: Any  # Callable that returns list[SubTask]
    default_complexity: str = "moderate"


def _create_setup_project(hint: str = "") -> list[SubTask]:
    """Decompose "setup/create a project" tasks."""
    return [
        SubTask(
            id="init-dirs",
            description=f"Create project directory structure{' for ' + hint if hint else ''}",
            dependencies=[],
            estimated_complexity="trivial",
            suggested_brain="action",
            input_hint="Project name and language/framework requirements",
            output_hint="Directory tree",
        ),
        SubTask(
            id="init-config",
            description="Create configuration files (pyproject.toml / package.json / etc.)",
            dependencies=["init-dirs"],
            estimated_complexity="simple",
            suggested_brain="action",
            input_hint="Project type and dependencies from user requirements",
            output_hint="Configuration files",
        ),
        SubTask(
            id="init-code",
            description="Write initial source code / entry point",
            dependencies=["init-config"],
            estimated_complexity="moderate",
            suggested_brain="skill",
            input_hint="Feature requirements from user",
            output_hint="Source code files",
        ),
        SubTask(
            id="init-tests",
            description="Write initial test cases",
            dependencies=["init-code"],
            estimated_complexity="moderate",
            suggested_brain="skill",
            input_hint="Source code to test",
            output_hint="Test files + test results",
        ),
        SubTask(
            id="verify",
            description="Run tests and verify project setup",
            dependencies=["init-tests"],
            estimated_complexity="simple",
            suggested_brain="action",
            input_hint="Test results",
            output_hint="Verification report",
        ),
    ]


def _create_analyze_fix(hint: str = "") -> list[SubTask]:
    """Decompose "analyze error logs / fix bug" tasks."""
    return [
        SubTask(
            id="gather-logs",
            description="Collect error logs, stack traces, and relevant context",
            dependencies=[],
            estimated_complexity="simple",
            suggested_brain="action",
            input_hint="Log file paths or error description",
            output_hint="Collected error data",
        ),
        SubTask(
            id="analyze-patterns",
            description="Analyze error patterns — root cause, frequency, affected components",
            dependencies=["gather-logs"],
            estimated_complexity="moderate",
            suggested_brain="skill",
            input_hint="Collected error data",
            output_hint="Root cause analysis",
        ),
        SubTask(
            id="locate-code",
            description="Locate the specific code/files causing the issue",
            dependencies=["analyze-patterns"],
            estimated_complexity="moderate",
            suggested_brain="action",
            input_hint="Root cause analysis",
            output_hint="List of affected files and line numbers",
        ),
        SubTask(
            id="apply-fix",
            description="Apply the fix to the identified code",
            dependencies=["locate-code"],
            estimated_complexity="complex",
            suggested_brain="skill",
            input_hint="Affected files + fix strategy",
            output_hint="Patched code changes",
        ),
        SubTask(
            id="verify-fix",
            description="Verify the fix: re-run tests, check logs, confirm resolution",
            dependencies=["apply-fix"],
            estimated_complexity="moderate",
            suggested_brain="action",
            input_hint="Patched code + original error context",
            output_hint="Verification results",
        ),
    ]


def _create_build_deploy(hint: str = "") -> list[SubTask]:
    """Decompose "build and deploy" tasks."""
    return [
        SubTask(
            id="lint",
            description="Run linters and static analysis",
            dependencies=[],
            estimated_complexity="simple",
            suggested_brain="action",
            input_hint="Source code path",
            output_hint="Lint results",
        ),
        SubTask(
            id="test",
            description="Run the full test suite",
            dependencies=["lint"],
            estimated_complexity="moderate",
            suggested_brain="action",
            input_hint="Test suite path",
            output_hint="Test results",
        ),
        SubTask(
            id="build",
            description="Build / compile the project",
            dependencies=["test"],
            estimated_complexity="moderate",
            suggested_brain="action",
            input_hint="Build configuration",
            output_hint="Build artifacts",
        ),
        SubTask(
            id="deploy",
            description="Deploy to target environment",
            dependencies=["build"],
            estimated_complexity="complex",
            suggested_brain="action",
            input_hint="Deployment target and credentials",
            output_hint="Deployment confirmation",
        ),
        SubTask(
            id="verify-deploy",
            description="Verify deployment: smoke tests, health checks",
            dependencies=["deploy"],
            estimated_complexity="simple",
            suggested_brain="action",
            input_hint="Deployment target URL",
            output_hint="Verification results",
        ),
    ]


def _create_implement_feature(hint: str = "") -> list[SubTask]:
    """Decompose "implement a feature" tasks."""
    return [
        SubTask(
            id="plan-design",
            description="Plan the feature design: architecture, components, interfaces",
            dependencies=[],
            estimated_complexity="moderate",
            suggested_brain="planning",
            input_hint="Feature requirements",
            output_hint="Design document",
        ),
        SubTask(
            id="implement-core",
            description="Implement the core logic / business rules",
            dependencies=["plan-design"],
            estimated_complexity="complex",
            suggested_brain="skill",
            input_hint="Design document",
            output_hint="Core implementation",
        ),
        SubTask(
            id="implement-tests",
            description="Write unit and integration tests",
            dependencies=["implement-core"],
            estimated_complexity="moderate",
            suggested_brain="skill",
            input_hint="Core implementation",
            output_hint="Test files",
        ),
        SubTask(
            id="document",
            description="Document the feature: API docs, usage examples",
            dependencies=["implement-core"],
            estimated_complexity="simple",
            suggested_brain="skill",
            input_hint="Feature implementation",
            output_hint="Documentation",
        ),
        SubTask(
            id="verify-feature",
            description="Run tests and validate against requirements",
            dependencies=["implement-tests", "document"],
            estimated_complexity="simple",
            suggested_brain="action",
            input_hint="Implementation + tests",
            output_hint="Verification report",
        ),
    ]


def _create_search_summarize(hint: str = "") -> list[SubTask]:
    """Decompose "search and summarize" tasks."""
    return [
        SubTask(
            id="search",
            description=f"Search for information{' about ' + hint if hint else ''}",
            dependencies=[],
            estimated_complexity="simple",
            suggested_brain="action",
            input_hint="Search query",
            output_hint="Raw search results",
        ),
        SubTask(
            id="gather",
            description="Gather and deduplicate results from all sources",
            dependencies=["search"],
            estimated_complexity="simple",
            suggested_brain="cheap",
            input_hint="Raw search results",
            output_hint="Deduplicated content",
        ),
        SubTask(
            id="analyze",
            description="Analyze gathered information for key insights",
            dependencies=["gather"],
            estimated_complexity="moderate",
            suggested_brain="skill",
            input_hint="Gathered content",
            output_hint="Key insights",
        ),
        SubTask(
            id="summarize",
            description="Write a structured summary with sources cited",
            dependencies=["analyze"],
            estimated_complexity="moderate",
            suggested_brain="skill",
            input_hint="Key insights",
            output_hint="Final summary",
        ),
    ]


def _create_data_processing(hint: str = "") -> list[SubTask]:
    """Decompose data processing / ETL tasks."""
    return [
        SubTask(
            id="extract",
            description="Extract data from source(s)",
            dependencies=[],
            estimated_complexity="moderate",
            suggested_brain="action",
            input_hint="Data source location / credentials",
            output_hint="Raw extracted data",
        ),
        SubTask(
            id="transform",
            description="Transform / clean / normalize the data",
            dependencies=["extract"],
            estimated_complexity="moderate",
            suggested_brain="skill",
            input_hint="Raw data + transformation rules",
            output_hint="Transformed data",
        ),
        SubTask(
            id="validate",
            description="Validate data integrity and quality",
            dependencies=["transform"],
            estimated_complexity="simple",
            suggested_brain="cheap",
            input_hint="Transformed data",
            output_hint="Validation report",
        ),
        SubTask(
            id="load",
            description="Load data into target destination",
            dependencies=["validate"],
            estimated_complexity="moderate",
            suggested_brain="action",
            input_hint="Validated data + target details",
            output_hint="Load confirmation",
        ),
    ]


def _create_document_generate(hint: str = "") -> list[SubTask]:
    """Decompose "write/generate/create document" tasks."""
    return [
        SubTask(
            id="research",
            description="Research the topic and gather reference material",
            dependencies=[],
            estimated_complexity="moderate",
            suggested_brain="action",
            input_hint="Topic and requirements",
            output_hint="Research notes + references",
        ),
        SubTask(
            id="outline",
            description="Create a structured outline / table of contents",
            dependencies=["research"],
            estimated_complexity="simple",
            suggested_brain="skill",
            input_hint="Research notes",
            output_hint="Structured outline",
        ),
        SubTask(
            id="draft",
            description="Write the first draft following the outline",
            dependencies=["outline"],
            estimated_complexity="complex",
            suggested_brain="skill",
            input_hint="Outline + research material",
            output_hint="First draft",
        ),
        SubTask(
            id="review",
            description="Review for accuracy, clarity, and completeness",
            dependencies=["draft"],
            estimated_complexity="moderate",
            suggested_brain="cheap",
            input_hint="First draft",
            output_hint="Review feedback",
        ),
        SubTask(
            id="finalize",
            description="Apply review feedback and produce final version",
            dependencies=["review"],
            estimated_complexity="simple",
            suggested_brain="skill",
            input_hint="Draft + review feedback",
            output_hint="Final document",
        ),
    ]


def _create_refactor(hint: str = "") -> list[SubTask]:
    """Decompose refactoring tasks."""
    return [
        SubTask(
            id="audit",
            description="Audit current codebase: structure, dependencies, quality metrics",
            dependencies=[],
            estimated_complexity="moderate",
            suggested_brain="action",
            input_hint="Codebase path",
            output_hint="Audit report",
        ),
        SubTask(
            id="plan-refactor",
            description="Plan refactoring strategy: what to change, in what order",
            dependencies=["audit"],
            estimated_complexity="moderate",
            suggested_brain="planning",
            input_hint="Audit report + refactoring goals",
            output_hint="Refactoring plan",
        ),
        SubTask(
            id="execute-refactor",
            description="Execute refactoring changes incrementally",
            dependencies=["plan-refactor"],
            estimated_complexity="complex",
            suggested_brain="skill",
            input_hint="Refactoring plan",
            output_hint="Refactored code",
        ),
        SubTask(
            id="verify-refactor",
            description="Run tests, check behavior parity, confirm no regressions",
            dependencies=["execute-refactor"],
            estimated_complexity="moderate",
            suggested_brain="action",
            input_hint="Refactored code + original tests",
            output_hint="Verification results",
        ),
    ]


# ── Template registry ───────────────────────────────────────────────────────


DECOMPOSITION_TEMPLATES: list[DecompositionTemplate] = [
    # Setup / Create project
    DecompositionTemplate(
        name="setup_project",
        patterns=[
            r"\b(set\s*up|setup|create|init|scaffold|start|new|generate)\b.*\b(project|repo|repository|app|application|package)\b",
            r"\b(project|repo|repository|app|application|package)\b.*\b(set\s*up|setup|create|init|scaffold|start)\b",
        ],
        keywords=[
            "setup project", "create project", "init project",
            "scaffold", "new project", "start project",
            "搭建项目", "创建项目", "新建项目", "初始化项目",
        ],
        subtask_factory=_create_setup_project,
    ),

    # Analyze error / Fix bug
    DecompositionTemplate(
        name="analyze_fix",
        patterns=[
            r"\b(debug|fix|resolve|troubleshoot|diagnose)\b.*\b(bug|error|issue|problem|failure|exception|crash)\b",
            r"\b(bug|error|issue|problem|failure|exception|crash)\b.*\b(debug|fix|resolve|troubleshoot)\b",
            r"\b(analy[sz]e)\b.*\b(error|log|crash|bug)\b",
            r"\b(error|log|crash|bug)\b.*\b(analy[sz]e|investigate)\b",
        ],
        keywords=[
            "fix bug", "debug", "fix error", "troubleshoot",
            "analyze error", "error analysis", "debug issue",
            "修复", "调试", "排查", "错误分析", "bug修复",
        ],
        subtask_factory=_create_analyze_fix,
    ),

    # Build and deploy
    DecompositionTemplate(
        name="build_deploy",
        patterns=[
            r"\b(build|compile|package)\b.*\b(deploy|release|ship|publish|launch)\b",
            r"\b(deploy|release|ship|publish|launch)\b.*\b(build|compile|package)\b",
            r"\b(deploy|release|ship|publish|launch)\b.*\b(prod|production|staging|server|cloud)\b",
        ],
        keywords=[
            "build and deploy", "deploy", "release", "ship",
            "publish", "launch", "ci/cd",
            "构建部署", "发布", "上线", "部署",
        ],
        subtask_factory=_create_build_deploy,
    ),

    # Implement feature
    DecompositionTemplate(
        name="implement_feature",
        patterns=[
            r"\b(implement|develop|write|code|build|add|create)\b.*\b(feature|functionality|module|component|endpoint)\b",
            r"\b(feature|functionality|module|component|endpoint)\b.*\b(implement|develop|write|code|build|add)\b",
        ],
        keywords=[
            "implement feature", "add feature", "develop feature",
            "write code for", "build module", "create component",
            "实现功能", "开发功能", "添加功能", "编写模块",
        ],
        subtask_factory=_create_implement_feature,
    ),

    # Search and summarize
    DecompositionTemplate(
        name="search_summarize",
        patterns=[
            r"\b(search|find|look\s*up|query)\b.*\b(summarize|summary|recap|brief|overview)\b",
            r"\b(summarize|summary|recap|brief|overview)\b.*\b(search|find|look\s*up)\b",
            r"\b(research|investigate)\b.*\b(report|summary|write\s*up)\b",
        ],
        keywords=[
            "search and summarize", "research and report",
            "find and summarize", "look up and summarize",
            "搜索总结", "调研报告", "查找并总结",
        ],
        subtask_factory=_create_search_summarize,
    ),

    # Data processing
    DecompositionTemplate(
        name="data_processing",
        patterns=[
            r"\b(process|transform|clean|normalize|migrate|convert)\b.*\b(data|dataset|csv|json|database|table)\b",
            r"\b(data|dataset|csv|json|database|table)\b.*\b(process|transform|clean|migrate|convert|import|export)\b",
            r"\b(etl|extract.*transform.*load)\b",
        ],
        keywords=[
            "process data", "transform data", "clean data",
            "data processing", "data pipeline", "etl",
            "数据处理", "数据清洗", "数据转换",
        ],
        subtask_factory=_create_data_processing,
    ),

    # Document generation
    DecompositionTemplate(
        name="document_generate",
        patterns=[
            r"\b(write|create|generate|compose|draft)\b.*\b(document|report|article|blog|readme|proposal|spec|guide|manual)\b",
            r"\b(document|report|article|blog|readme|proposal|spec|guide|manual)\b.*\b(write|create|generate|compose|draft)\b",
        ],
        keywords=[
            "write document", "create report", "generate report",
            "write article", "draft proposal",
            "写文档", "写报告", "起草", "生成文档",
        ],
        subtask_factory=_create_document_generate,
    ),

    # Refactor
    DecompositionTemplate(
        name="refactor",
        patterns=[
            r"\b(refactor|restructure|reorganize|rewrite|clean\s*up)\b.*\b(code|codebase|module|class|function)\b",
            r"\b(code|codebase|module|class|function)\b.*\b(refactor|restructure|reorganize|rewrite|clean\s*up)\b",
        ],
        keywords=[
            "refactor", "restructure code", "clean up code",
            "rewrite code", "code cleanup",
            "重构", "代码清理", "代码重构",
        ],
        subtask_factory=_create_refactor,
    ),
]


# ── Multi-step detection patterns ────────────────────────────────────────────


# Keywords that suggest a task has multiple steps
MULTI_STEP_INDICATORS: list[str] = [
    # Sequential connectors
    "then", "after that", "next", "finally", "first",
    "second", "third", "step 1", "step 2", "step 3",
    # Composite action words
    "and also", "as well as", "while", "meanwhile",
    # Chinese connectors
    "然后", "接着", "之后", "最后", "首先",
    "第一步", "第二步", "第三步", "然后还要",
    "同时", "另外",
]


# ── PlanningBrain ───────────────────────────────────────────────────────────


class PlanningBrain(BaseBrain):
    """Task decomposition and planning brain.

    Takes complex tasks and breaks them down into ordered, dependency-aware
    subtask DAGs. In Phase 1 this is purely rule-based using template matching.
    Phase 2 will add LLM-powered decomposition via the API Manager.

    Usage::

        brain = PlanningBrain()
        request = BrainRequest(
            task_id="t1",
            user_input="setup a new python project with tests",
            complexity=TaskComplexity.MODERATE,
        )
        response = await brain.process(request)
        plan = response.structured_output  # Plan dataclass
    """

    @property
    def brain_type(self) -> BrainType:
        return BrainType.PLANNING

    # ── Public API ──────────────────────────────────────────────────────

    async def process(self, request: BrainRequest) -> BrainResponse:
        """Execute the planning pipeline.

        Stages:
            1. Analyze task → extract goal, constraints, user intent
            2. Match against template library
            3. Build subtask DAG + topological sort
            4. Return structured Plan

        Falls back to API-mode if an api_manager is provided in metadata
        and no rule template matches.
        """
        text = request.user_input.strip()
        if not text:
            return BrainResponse(
                success=False,
                content="Empty input — cannot plan.",
                brain_type=BrainType.PLANNING,
                confidence=0.0,
                errors=["empty_input"],
            )

        # Stage 1: Match against rule templates
        matched_template, confidence = self._match_template(text)
        hint = self._extract_hint(text, matched_template)

        if matched_template is not None:
            # Rule-based decomposition
            plan = self._build_plan(
                request=request,
                template=matched_template,
                hint=hint,
                confidence=confidence,
                plan_type="rule",
            )
        else:
            # Try API-based decomposition if api_manager is available
            api_manager = request.metadata.get("api_manager")
            if api_manager:
                try:
                    plan = await self._decompose_via_llm(
                        request=request,
                        api_manager=api_manager,
                    )
                except Exception as exc:
                    # Fallback to generic decomposition
                    plan = self._build_generic_plan(request=request, error=str(exc))
            else:
                # Generic fallback — minimal decomposition
                plan = self._build_generic_plan(request=request)

        # Build response
        content = self._format_plan_summary(plan)

        return BrainResponse(
            success=True,
            content=content,
            brain_type=BrainType.PLANNING,
            tokens_used=0,
            cost_usd=0.0,
            latency_ms=0.0,
            confidence=confidence,
            suggested_next_brain=(
                BrainType.ACTION
                if plan.estimated_total_complexity in ("simple", "moderate")
                else BrainType.SKILL
            ),
            structured_output=plan,
            metadata={
                "brain_version": "1.0.0",
                "plan_type": plan.plan_type,
                "subtask_count": len(plan.subtasks),
                "execution_order": plan.execution_order,
            },
        )

    def can_handle(self, request: BrainRequest) -> bool:
        """Determine if this task needs planning.

        Returns True when:
        - TaskComplexity is MODERATE or higher
        - Input contains multi-step indicators
        - Input length suggests a complex task (> 80 chars)
        - Input matches a known decomposition template
        """
        # Complexity-based
        if request.complexity is not None:
            if request.complexity.value >= TaskComplexity.MODERATE.value:
                return True

        # Multi-step indicators in the input
        text_lower = request.user_input.lower()
        if any(indicator in text_lower for indicator in MULTI_STEP_INDICATORS):
            return True

        # Long input suggests complexity
        if len(request.user_input) > 80:
            return True

        # Matches a known template
        matched, _ = self._match_template(request.user_input)
        if matched is not None:
            return True

        return False

    # ── Template matching ────────────────────────────────────────────────

    def _match_template(
        self, text: str
    ) -> tuple[Optional[DecompositionTemplate], float]:
        """Match user input against the decomposition template library.

        Returns:
            Tuple of (matched_template, confidence_score).
            confidence ranges 0.0 → 1.0.
        """
        text_lower = text.lower()
        best_template: Optional[DecompositionTemplate] = None
        best_score: float = 0.0

        for template in DECOMPOSITION_TEMPLATES:
            score = 0.0

            # Keyword matching (weight: 0.5)
            if template.keywords:
                hits = sum(
                    1 for kw in template.keywords if kw.lower() in text_lower
                )
                if hits > 0:
                    # Normalize: more hits → higher score
                    keyword_score = min(hits / max(len(template.keywords) * 0.3, 1), 1.0)
                    score += keyword_score * 0.5

            # Regex pattern matching (weight: 0.5)
            if template.patterns:
                pattern_hits = 0
                for pat in template.patterns:
                    try:
                        if re.search(pat, text, re.IGNORECASE):
                            pattern_hits += 1
                    except re.error:
                        continue
                if pattern_hits > 0:
                    pattern_score = min(pattern_hits / max(len(template.patterns) * 0.5, 1), 1.0)
                    score += pattern_score * 0.5

            if score > best_score:
                best_score = score
                best_template = template

        if best_template is None or best_score < 0.2:
            return None, 0.0

        return best_template, round(min(best_score, 1.0), 3)

    def _extract_hint(
        self, text: str, template: Optional[DecompositionTemplate]
    ) -> str:
        """Extract a contextual hint from the user's input for subtask descriptions.

        Tries to identify the specific subject/project/technology mentioned.
        """
        if template is None:
            return ""

        # Remove known template keywords/patterns to isolate the subject
        cleaned = text
        for kw in template.keywords:
            cleaned = cleaned.replace(kw, "")
        for pat in template.patterns:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)

        # Clean up delimiters and whitespace
        cleaned = re.sub(r"[.,;:!?\n]+", " ", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

        # Cap hint length
        return cleaned[:80] if cleaned else ""

    # ── Plan construction ────────────────────────────────────────────────

    def _build_plan(
        self,
        request: BrainRequest,
        template: DecompositionTemplate,
        hint: str,
        confidence: float,
        plan_type: str,
    ) -> Plan:
        """Build a Plan from a matched template.

        Args:
            request: Original BrainRequest.
            template: Matched decomposition template.
            hint: Extracted context hint.
            confidence: Match confidence score.
            plan_type: "rule" or "llm".

        Returns:
            Structured Plan with subtasks and execution order.
        """
        # Generate subtasks from the template factory
        subtasks = template.subtask_factory(hint)

        # Topological sort to determine execution order
        execution_order = self._topological_sort(subtasks)

        # Estimate total complexity
        total_complexity = self._estimate_total_complexity(subtasks)

        return Plan(
            task_id=request.task_id,
            original_input=request.user_input,
            subtasks=subtasks,
            execution_order=execution_order,
            estimated_total_complexity=total_complexity,
            plan_type=plan_type,
            metadata={
                "template_name": template.name,
                "confidence": confidence,
                "hint": hint,
            },
        )

    def _build_generic_plan(
        self,
        request: BrainRequest,
        error: str = "",
    ) -> Plan:
        """Build a minimal generic plan when no template matches.

        This is a safety net: a 3-step plan (understand → execute → verify).
        """
        subtasks = [
            SubTask(
                id="understand",
                description="Clarify and understand the task requirements",
                dependencies=[],
                estimated_complexity="simple",
                suggested_brain="cheap",
                input_hint=request.user_input,
                output_hint="Clarified requirements",
            ),
            SubTask(
                id="execute",
                description=f"Execute: {request.user_input[:120]}",
                dependencies=["understand"],
                estimated_complexity="moderate",
                suggested_brain="skill",
                input_hint="Clarified requirements",
                output_hint="Execution results",
            ),
            SubTask(
                id="verify",
                description="Verify results and confirm completion",
                dependencies=["execute"],
                estimated_complexity="simple",
                suggested_brain="cheap",
                input_hint="Execution results",
                output_hint="Verification confirmation",
            ),
        ]

        return Plan(
            task_id=request.task_id,
            original_input=request.user_input,
            subtasks=subtasks,
            execution_order=["understand", "execute", "verify"],
            estimated_total_complexity="moderate",
            plan_type="generic",
            metadata={
                "fallback_reason": error or "no matching template",
            },
        )

    async def _decompose_via_llm(
        self,
        request: BrainRequest,
        api_manager: Any,
    ) -> Plan:
        """Decompose a task using an LLM via the API Manager.

        Sends a structured planning prompt to the LLM and parses the JSON
        response into a Plan with SubTask DAG. On JSON parse failure, falls
        back to rule-based template matching.

        Args:
            request: Original BrainRequest.
            api_manager: APIManager instance from metadata.

        Returns:
            Structured Plan.

        Raises:
            ValueError: If LLM response is unparseable AND no template matches.
        """
        # Build messages in OpenAI chat format
        system_prompt = self._build_llm_planning_prompt(request)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.user_input},
        ]

        # Determine which API to use
        api_name = request.metadata.get("api_name")
        if api_name is None:
            try:
                api_name = api_manager.get_most_capable().name
            except Exception:
                api_name = api_manager.list_apis()[0].name if api_manager.list_apis() else "default"  # type: ignore[union-attr]

        # Call the LLM
        raw_response: dict[str, Any] = await api_manager.call(
            api_name=api_name,
            messages=messages,
            temperature=0.2,
            max_tokens=min(request.max_tokens or 2048, 4096),
        )

        # Extract content and usage from OpenAI-compatible response
        choices: list[dict[str, Any]] = raw_response.get("choices", [])
        if not choices:
            raise ValueError(f"LLM returned empty choices: {raw_response}")
        llm_content: str = choices[0].get("message", {}).get("content", "")
        usage: dict[str, int] = raw_response.get("usage", {})

        # Resolve API config for cost calculation
        try:
            api_config = api_manager.get_api(api_name)
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
            cost_usd = round(
                (input_tokens / 1000) * api_config.price_per_1k_input
                + (output_tokens / 1000) * api_config.price_per_1k_output,
                6,
            )
        except Exception:
            total_tokens = usage.get("total_tokens", 0)
            cost_usd = 0.0

        # Parse JSON — on failure, fallback to rule templates
        try:
            plan = self._parse_llm_response(request, llm_content)
        except (ValueError, json.JSONDecodeError) as parse_error:
            # Fallback: try rule-based template matching
            matched_template, confidence = self._match_template(request.user_input)
            if matched_template is not None:
                hint = self._extract_hint(request.user_input, matched_template)
                plan = self._build_plan(
                    request=request,
                    template=matched_template,
                    hint=hint,
                    confidence=confidence,
                    plan_type="rule",
                )
                plan.metadata["llm_fallback_reason"] = (
                    f"JSON parse failed: {parse_error}; fell back to "
                    f"template '{matched_template.name}'"
                )
            else:
                plan = self._build_generic_plan(
                    request=request,
                    error=f"LLM JSON parse failed and no template matched: {parse_error}",
                )

        # Record LLM call cost in plan metadata
        plan.metadata["llm_api_name"] = api_name
        plan.metadata["llm_tokens_input"] = usage.get("prompt_tokens", 0)
        plan.metadata["llm_tokens_output"] = usage.get("completion_tokens", 0)
        plan.metadata["llm_tokens_total"] = total_tokens
        plan.metadata["llm_cost_usd"] = cost_usd

        return plan

    def _build_llm_planning_prompt(self, request: BrainRequest) -> str:
        """Build the system prompt for LLM-based task decomposition."""
        # Include compiled context as additional guidance when available
        context_hint = ""
        if request.compiled_context:
            context_hint = f"\n\nAdditional context: {request.compiled_context}"

        return f"""You are a task planning expert. Decompose the following task into subtasks.
Return ONLY a JSON array of objects with keys:
- id (string): unique subtask id
- description (string): what this step does
- dependencies (list of ids): which subtasks must complete first
- estimated_complexity (string): trivial/simple/moderate/complex
- suggested_brain (string): cheap/skill/action/memory/planning/frontier
{context_hint}"""

    def _parse_llm_response(self, request: BrainRequest, llm_text: str) -> Plan:
        """Parse LLM response text into a structured Plan.

        Args:
            request: Original BrainRequest.
            llm_text: Raw text content from the LLM response.

        Returns:
            Structured Plan.

        Raises:
            ValueError: If the text cannot be parsed as valid JSON.
        """
        import json

        # Strip markdown code fences if present
        text = llm_text.strip()
        if text.startswith("```"):
            # Remove opening fence
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            # Remove closing fence
            text = re.sub(r"\n?```\s*$", "", text)

        try:
            raw_tasks = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON array
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                raw_tasks = json.loads(match.group(0))
            else:
                raise ValueError(f"Could not parse LLM response as JSON: {text[:200]}")

        subtasks = [
            SubTask(
                id=t["id"],
                description=t["description"],
                dependencies=t.get("dependencies", []),
                estimated_complexity=t.get("estimated_complexity", "simple"),
                suggested_brain=t.get("suggested_brain", "cheap"),
                input_hint=t.get("input_hint", ""),
                output_hint=t.get("output_hint", ""),
            )
            for t in raw_tasks
        ]

        execution_order = self._topological_sort(subtasks)

        return Plan(
            task_id=request.task_id,
            original_input=request.user_input,
            subtasks=subtasks,
            execution_order=execution_order,
            estimated_total_complexity=self._estimate_total_complexity(subtasks),
            plan_type="llm",
            metadata={
                "decomposition_method": "llm",
                "llm_response_length": len(text),
            },
        )

    # ── DAG utilities ────────────────────────────────────────────────────

    @staticmethod
    def _topological_sort(subtasks: list[SubTask]) -> list[str]:
        """Perform Kahn's algorithm for topological sorting of subtasks.

        Returns subtask IDs in execution order. If the DAG has a cycle,
        returns IDs in the best-effort order (with cycle detection).

        Args:
            subtasks: List of SubTask objects forming a DAG.

        Returns:
            List of subtask IDs in valid execution order.
        """
        # Build adjacency and in-degree
        in_degree: dict[str, int] = {}
        adjacency: dict[str, list[str]] = {}

        for st in subtasks:
            if st.id not in in_degree:
                in_degree[st.id] = 0
            adjacency.setdefault(st.id, [])

        for st in subtasks:
            for dep in st.dependencies:
                in_degree[st.id] = in_degree.get(st.id, 0) + 1
                adjacency.setdefault(dep, []).append(st.id)
                # Ensure dependency nodes exist in in_degree
                if dep not in in_degree:
                    in_degree[dep] = 0

        # Kahn's algorithm
        queue: deque[str] = deque(
            node for node, deg in in_degree.items() if deg == 0
        )
        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in adjacency.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Cycle detection: if not all nodes processed, there's a cycle
        if len(result) != len(in_degree):
            missing = [n for n in in_degree if n not in result]
            result.extend(missing)

        return result

    @staticmethod
    def _estimate_total_complexity(subtasks: list[SubTask]) -> str:
        """Estimate overall plan complexity from subtask complexities.

        Uses a weighted heuristic based on the highest individual complexity
        and the number of tasks.

        Returns:
            "trivial" | "simple" | "moderate" | "complex" | "highly_complex"
        """
        if not subtasks:
            return "simple"

        complexity_weights = {
            "trivial": 1,
            "simple": 2,
            "moderate": 3,
            "complex": 4,
            "highly_complex": 5,
        }

        max_complexity = 0
        total_weight = 0
        for st in subtasks:
            w = complexity_weights.get(st.estimated_complexity, 2)
            max_complexity = max(max_complexity, w)
            total_weight += w

        avg_weight = total_weight / len(subtasks)

        # Combine max and average with task count
        score = max_complexity * 0.6 + avg_weight * 0.3 + min(len(subtasks) * 0.1, 1.0)

        if score <= 1.5:
            return "trivial"
        elif score <= 2.5:
            return "simple"
        elif score <= 3.5:
            return "moderate"
        elif score <= 4.5:
            return "complex"
        else:
            return "highly_complex"

    # ── Response formatting ──────────────────────────────────────────────

    @staticmethod
    def _format_plan_summary(plan: Plan) -> str:
        """Format a human-readable summary of the plan.

        Args:
            plan: The structured Plan object.

        Returns:
            A multi-line string describing the plan.
        """
        lines = [
            f"[PlanningBrain] Decomposed into {len(plan.subtasks)} subtasks",
            f"  Type: {plan.plan_type.upper()}",
            f"  Estimated complexity: {plan.estimated_total_complexity}",
            f"  Execution order: {' → '.join(plan.execution_order)}",
            "",
        ]

        # Dependency graph
        lines.append("  Dependency DAG:")
        for st in plan.subtasks:
            deps = st.dependencies if st.dependencies else ["(none)"]
            lines.append(
                f"    [{st.id}] ({st.estimated_complexity}) {st.description[:80]}"
            )
            lines.append(f"      Depends on: {', '.join(deps)}")
            lines.append(f"      Suggested brain: {st.suggested_brain}")

        return "\n".join(lines)

    # ── Public helpers ──────────────────────────────────────────────────

    def decompose(self, text: str) -> Optional[Plan]:
        """Public: quick decomposition without a full BrainRequest.

        Args:
            text: Task description.

        Returns:
            Plan if a template matches, None otherwise.
        """
        matched, confidence = self._match_template(text)
        if matched is None:
            return None

        hint = self._extract_hint(text, matched)
        import uuid
        return self._build_plan(
            request=BrainRequest(task_id=uuid.uuid4().hex[:12], user_input=text),
            template=matched,
            hint=hint,
            confidence=confidence,
            plan_type="rule",
        )

    @staticmethod
    def list_templates() -> list[str]:
        """List all available decomposition template names."""
        return [t.name for t in DECOMPOSITION_TEMPLATES]

    @staticmethod
    def template_info(name: str) -> Optional[dict[str, Any]]:
        """Get detailed info about a specific template.

        Args:
            name: Template name.

        Returns:
            Dict with name, patterns, keywords, subtask_count, or None.
        """
        for t in DECOMPOSITION_TEMPLATES:
            if t.name == name:
                dummy_tasks = t.subtask_factory("")
                return {
                    "name": t.name,
                    "patterns": t.patterns,
                    "keywords": t.keywords,
                    "subtask_count": len(dummy_tasks),
                    "default_complexity": t.default_complexity,
                }
        return None
