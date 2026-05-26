"""
Unified verification layer for Octopus brain outputs.

Every BrainResponse passes through the Verifier before reaching the user.
The verifier inspects code safety, factual consistency, and output quality,
producing a structured VerificationResult with actionable suggestions.

Architecture:
    Verifier (orchestrator)
    ├── CodeVerifier  — Python ast.parse, dangerous-call detection
    └── FactVerifier  — Hallucination signals, contradiction detection
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from ..brains.base import BrainResponse, BrainType


# ── Severity ────────────────────────────────────────────────────────────────


class Severity(str, Enum):
    """Check severity — drives whether the response passes verification."""

    INFO = "info"          # FYI, always non-blocking
    WARNING = "warning"    # May indicate a problem, doesn't block
    ERROR = "error"        # Concrete issue, blocks passing
    CRITICAL = "critical"  # Must-fix, always blocks


# ── Result ───────────────────────────────────────────────────────────────────


@dataclass
class VerificationResult:
    """Structured result after verification completes.

    Attributes:
        passed: True iff no ERROR or CRITICAL checks failed.
        checks: Per-check details (name, passed, detail, severity).
        errors: Flat list of error strings for logging/display.
        warnings: Flat list of warning strings.
        suggestion: Human-readable remediation advice when failed.
    """

    passed: bool = True
    checks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestion: str = ""

    def merge(self, other: "VerificationResult") -> "VerificationResult":
        """Combine another result into this one (mutates & returns self)."""
        self.passed = self.passed and other.passed
        self.checks.extend(other.checks)
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if other.suggestion and not self.suggestion:
            self.suggestion = other.suggestion
        elif other.suggestion:
            self.suggestion = f"{self.suggestion}; {other.suggestion}"
        return self

    def add_check(
        self,
        name: str,
        passed: bool,
        detail: str = "",
        severity: Severity = Severity.ERROR,
    ) -> None:
        """Append a single check with proper severity handling."""
        entry: dict[str, Any] = {
            "name": name,
            "passed": passed,
            "detail": detail,
            "severity": severity.value,
        }
        self.checks.append(entry)

        if passed:
            return

        if severity in (Severity.ERROR, Severity.CRITICAL):
            self.passed = False
            self.errors.append(f"[{name}] {detail}")
        elif severity == Severity.WARNING:
            self.warnings.append(f"[{name}] {detail}")
        # INFO failures are logged as checks but don't emit warnings


# ── Dangerous Call Registry ──────────────────────────────────────────────────

# Functions / builtins considered dangerous in untrusted code generation.
# Each entry: (fully-qualified-name, risk_description)
DANGEROUS_CALLS: dict[str, str] = {
    "eval": "eval() allows arbitrary code execution",
    "exec": "exec() allows arbitrary code execution",
    "compile": "compile() can execute arbitrary code objects",
    "__import__": "__import__() bypasses module-level restrictions",
    "open": "open() may read/write arbitrary files",
}

DANGEROUS_MODULES: dict[str, str] = {
    "os.system": "os.system() spawns arbitrary shell commands",
    "os.popen": "os.popen() opens a pipe to an arbitrary command",
    "os.spawnl": "os.spawn*() family executes arbitrary processes",
    "os.spawnle": "os.spawn*() family executes arbitrary processes",
    "os.spawnlp": "os.spawn*() family executes arbitrary processes",
    "os.spawnlpe": "os.spawn*() family executes arbitrary processes",
    "os.spawnv": "os.spawn*() family executes arbitrary processes",
    "os.spawnve": "os.spawn*() family executes arbitrary processes",
    "os.spawnvp": "os.spawn*() family executes arbitrary processes",
    "os.spawnvpe": "os.spawn*() family executes arbitrary processes",
    "os.execv": "os.exec*() replaces current process",
    "os.execve": "os.exec*() replaces current process",
    "os.execl": "os.exec*() replaces current process",
    "os.execle": "os.exec*() replaces current process",
    "os.execlp": "os.exec*() replaces current process",
    "os.execlpe": "os.exec*() replaces current process",
    "subprocess.call": "subprocess.call() executes arbitrary commands",
    "subprocess.run": "subprocess.run() executes arbitrary commands",
    "subprocess.Popen": "subprocess.Popen() spawns arbitrary processes",
    "subprocess.check_output": "subprocess.check_output() executes arbitrary commands",
    "subprocess.check_call": "subprocess.check_call() executes arbitrary commands",
    "shutil.rmtree": "shutil.rmtree() recursively deletes directories",
    "os.remove": "os.remove() deletes files",
    "os.unlink": "os.unlink() deletes files",
    "os.rmdir": "os.rmdir() removes directories",
    "ctypes.CDLL": "ctypes.CDLL loads arbitrary native code",
    "ctypes.WinDLL": "ctypes.WinDLL loads arbitrary native code",
}


# ── Code Verifier ────────────────────────────────────────────────────────────


class CodeVerifier:
    """Verifies generated Python code for syntax and safety issues.

    Phase 1 capabilities:
        - ast.parse() for syntax validation
        - Detection of dangerous builtins (eval, exec, etc.)
        - Detection of dangerous module calls (os.system, subprocess, etc.)
        - Detection of common pitfalls (bare except, mutable defaults)

    Usage:
        result = CodeVerifier().verify(code, language="python")
    """

    SUPPORTED_LANGUAGES = {"python", "py"}

    def verify(self, code: str, language: str = "python") -> VerificationResult:
        """Run all code checks against the given source.

        Args:
            code: Source code string to verify.
            language: Programming language (only "python"/"py" supported).

        Returns:
            VerificationResult with syntax and safety findings.
        """
        if language.lower() not in self.SUPPORTED_LANGUAGES:
            return VerificationResult(
                passed=True,
                checks=[{
                    "name": "language_skip",
                    "passed": True,
                    "detail": f"Skipping CodeVerifier for unsupported language: {language}",
                    "severity": Severity.INFO.value,
                }],
                suggestion="",
            )

        result = VerificationResult()

        # 1. Syntax check
        tree = self._check_syntax(code, result)

        # 2. AST-level safety checks (only if parsing succeeded)
        if tree is not None:
            self._check_dangerous_calls(tree, code, result)
            self._check_code_quality(tree, code, result)

        # 3. Build suggestion
        result.suggestion = self._build_suggestion(result)

        return result

    # ── individual checks ────────────────────────────────────────────────

    def _check_syntax(
        self, code: str, result: VerificationResult
    ) -> Optional[ast.Module]:
        """Parse code with ast; return the tree or None on failure."""
        try:
            tree = ast.parse(code)
            result.add_check(
                name="syntax",
                passed=True,
                detail="Code parsed successfully.",
                severity=Severity.INFO,
            )
            return tree
        except SyntaxError as e:
            line = e.lineno or "?"
            result.add_check(
                name="syntax",
                passed=False,
                detail=f"Line {line}: {e.msg}",
                severity=Severity.ERROR,
            )
            return None

    def _check_dangerous_calls(
        self, tree: ast.Module, code: str, result: VerificationResult
    ) -> None:
        """Walk the AST and detect calls to dangerous builtins / modules."""

        class DangerousVisitor(ast.NodeVisitor):
            def __init__(self):
                self.findings: list[tuple[int, str, str]] = []  # (line, name, detail)

            def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
                # Direct function call: foo()
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                    if name in DANGEROUS_CALLS:
                        self.findings.append(
                            (node.lineno, name, DANGEROUS_CALLS[name])
                        )

                # Attribute call: os.system(), subprocess.run()
                elif isinstance(node.func, ast.Attribute):
                    attr_chain = self._resolve_attr(node.func)
                    if attr_chain:
                        full = ".".join(attr_chain)
                        for danger, desc in DANGEROUS_MODULES.items():
                            # Match exactly or match suffix (e.g. os.path → not dangerous)
                            if full == danger or full.endswith("." + danger):
                                self.findings.append(
                                    (node.lineno, full, desc)
                                )
                                break

                self.generic_visit(node)

            @staticmethod
            def _resolve_attr(node: ast.Attribute) -> Optional[list[str]]:
                """Resolve a.b.c → ['a', 'b', 'c']."""
                parts = [node.attr]
                current = node.value
                while isinstance(current, ast.Attribute):
                    parts.insert(0, current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.insert(0, current.id)
                    return parts
                return None  # e.g. obj.something() — can't resolve

        visitor = DangerousVisitor()
        visitor.visit(tree)

        if visitor.findings:
            for line, name, detail in visitor.findings:
                result.add_check(
                    name=f"dangerous_call:{name}",
                    passed=False,
                    detail=f"Line {line}: {name} — {detail}",
                    severity=Severity.WARNING,
                )
        else:
            result.add_check(
                name="dangerous_calls",
                passed=True,
                detail="No dangerous calls detected.",
                severity=Severity.INFO,
            )

    def _check_code_quality(
        self, tree: ast.Module, code: str, result: VerificationResult
    ) -> None:
        """Catch common Python pitfalls."""
        bare_excepts: list[int] = []
        mutable_defaults: list[tuple[int, str]] = []

        class QualityVisitor(ast.NodeVisitor):
            def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:  # noqa: N802
                # bare except: without exception type
                if node.type is None:
                    bare_excepts.append(node.lineno)
                self.generic_visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        mutable_defaults.append(
                            (node.lineno, f"mutable default argument in '{node.name}'")
                        )
                        break  # one finding per function is enough
                self.generic_visit(node)

        QualityVisitor().visit(tree)

        for lineno in bare_excepts:
            result.add_check(
                name="bare_except",
                passed=False,
                detail=f"Line {lineno}: bare 'except:' — consider specifying exception types.",
                severity=Severity.WARNING,
            )

        for lineno, msg in mutable_defaults:
            result.add_check(
                name="mutable_default",
                passed=False,
                detail=f"Line {lineno}: {msg}",
                severity=Severity.WARNING,
            )

        if not bare_excepts and not mutable_defaults:
            result.add_check(
                name="code_quality",
                passed=True,
                detail="No common code quality issues detected.",
                severity=Severity.INFO,
            )

    def _build_suggestion(self, result: VerificationResult) -> str:
        """Generate a human-readable suggestion from findings."""
        if result.passed and not result.warnings:
            return ""
        parts: list[str] = []
        if result.errors:
            parts.append(
                f"Fix {len(result.errors)} error(s) first: "
                + "; ".join(result.errors)
            )
        if result.warnings:
            parts.append(
                f"Consider addressing {len(result.warnings)} warning(s): "
                + "; ".join(result.warnings)
            )
        return " ".join(parts)


# ── Fact Verifier ────────────────────────────────────────────────────────────


class FactVerifier:
    """Lightweight factual consistency checker (Phase 1).

    Phase 1 capabilities (heuristic, no external knowledge base):
        - Over-confident ignorance detection ("I don't know, but here's …")
        - Self-contradiction: same text containing pairs of opposing claims
        - Confidence marker consistency check
        - Hedge ratio: excessive hedges ("might", "could", "possibly")

    Phase 2 (roadmap): external source grounding, fact-check API integration.
    """

    # Pairs of opposing words/phrases used for contradiction detection.
    # Each pair is (positive-claim, negative-claim).
    OPPOSITION_PAIRS: list[tuple[str, str]] = [
        (r"\bis\b", r"\bis not\b"),
        (r"\bcan\b", r"\bcannot\b"),
        (r"\bwill\b", r"\bwill not\b"),
        (r"\bdoes\b", r"\bdoes not\b"),
        (r"\bhas\b", r"\bhas no\b"),
        (r"\balways\b", r"\bnever\b"),
        (r"\ball\b", r"\bnone\b"),
        (r"\btrue\b", r"\bfalse\b"),
        (r"\byes\b", r"\bno\b"),
        (r"\bcorrect\b", r"\bincorrect\b"),
        (r"\bsupported\b", r"\bunsupported\b"),
    ]

    # Over-confident ignorance patterns.
    IGNORANCE_PATTERNS: list[re.Pattern] = [
        re.compile(
            r"(i\s+don'?t\s+know|i\s+am\s+not\s+sure|i\s+cannot\s+(confirm|verify|find|access|determine))"
            r".*?(but|however|although|here'?s|let me|nonetheless)",
            re.IGNORECASE,
        ),
    ]

    # Excessive hedge words (ratio check).
    HEDGE_WORDS: list[str] = [
        "might", "may", "could", "possibly", "perhaps",
        "likely", "probably", "seems", "appears", "tends to",
        "generally", "typically", "often", "sometimes",
    ]

    # Confidence markers that should be self-consistent.
    CONFIDENCE_HIGH = re.compile(
        r"\b(definitely|certainly|absolutely|without (a |any )?doubt|undoubtedly|surely)\b",
        re.IGNORECASE,
    )
    CONFIDENCE_LOW = re.compile(
        r"\b(might|may|could|possibly|perhaps|maybe|unlikely|uncertain|not sure|unclear)\b",
        re.IGNORECASE,
    )

    def verify(self, text: str, context: Optional[Any] = None) -> VerificationResult:
        """Run all Phase 1 fact checks against the given text.

        Args:
            text: The LLM-generated text to verify.
            context: Optional additional context (reserved for Phase 2).

        Returns:
            VerificationResult with factual consistency findings.
        """
        result = VerificationResult()

        # Guard: empty or very short text
        if not text or len(text.strip()) < 10:
            result.add_check(
                name="fact_empty",
                passed=True,
                detail="Text too short for meaningful fact checks.",
                severity=Severity.INFO,
            )
            return result

        self._check_ignorance_pattern(text, result)
        self._check_contradictions(text, result)
        self._check_confidence_consistency(text, result)
        self._check_hedge_ratio(text, result)

        result.suggestion = self._build_suggestion(result)
        return result

    # ── individual checks ────────────────────────────────────────────────

    def _check_ignorance_pattern(
        self, text: str, result: VerificationResult
    ) -> None:
        """Detect 'I don't know, but here's a guess…' patterns."""
        found = False
        for pat in self.IGNORANCE_PATTERNS:
            match = pat.search(text)
            if match:
                found = True
                snippet = text[max(0, match.start() - 20): match.end() + 40]
                result.add_check(
                    name="overconfident_ignorance",
                    passed=False,
                    detail=(
                        "Admits uncertainty then provides answer anyway: "
                        f'"{snippet.strip()}..."'
                    ),
                    severity=Severity.WARNING,
                )
        if not found:
            result.add_check(
                name="ignorance_check",
                passed=True,
                detail="No over-confident ignorance patterns detected.",
                severity=Severity.INFO,
            )

    def _check_contradictions(
        self, text: str, result: VerificationResult
    ) -> None:
        """Detect self-contradictions within the same text."""
        # Split into sentences for pair-wise checking
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) < 2:
            result.add_check(
                name="contradictions",
                passed=True,
                detail="Single sentence — contradiction check skipped.",
                severity=Severity.INFO,
            )
            return

        contradictions: list[str] = []
        for i, sent_a in enumerate(sentences):
            for j, sent_b in enumerate(sentences):
                if i >= j:
                    continue
                for pos_pat, neg_pat in self.OPPOSITION_PAIRS:
                    # Check if sentence A says X and sentence B says not-X
                    # around the same subject
                    pos_a = re.search(pos_pat, sent_a, re.IGNORECASE)
                    neg_b = re.search(neg_pat, sent_b, re.IGNORECASE)
                    pos_b = re.search(pos_pat, sent_b, re.IGNORECASE)
                    neg_a = re.search(neg_pat, sent_a, re.IGNORECASE)

                    if (pos_a and neg_b) or (neg_a and pos_b):
                        # Extract the subject word near the match
                        ctx_a = sent_a[:100].strip()
                        ctx_b = sent_b[:100].strip()
                        contradiction = (
                            f'"{ctx_a}..." vs "{ctx_b}..." '
                            f"({pos_pat.pattern} / {neg_pat.pattern})"
                        )
                        if contradiction not in contradictions:
                            contradictions.append(contradiction)

        if contradictions:
            for i, c in enumerate(contradictions[:3]):  # cap at 3 for readability
                result.add_check(
                    name=f"contradiction_{i+1}",
                    passed=False,
                    detail=f"Possible self-contradiction: {c}",
                    severity=Severity.WARNING,
                )
        else:
            result.add_check(
                name="contradictions",
                passed=True,
                detail="No self-contradictions detected.",
                severity=Severity.INFO,
            )

    def _check_confidence_consistency(
        self, text: str, result: VerificationResult
    ) -> None:
        """Check that confidence markers don't mix high/low in same statement."""
        high_matches = self.CONFIDENCE_HIGH.findall(text)
        low_matches = self.CONFIDENCE_LOW.findall(text)

        if high_matches and low_matches:
            result.add_check(
                name="confidence_mix",
                passed=False,
                detail=(
                    f"Mixed confidence signals: high ({', '.join(set(high_matches[:3]))}) "
                    f"+ low ({', '.join(set(low_matches[:3]))})"
                ),
                severity=Severity.INFO,
            )
        else:
            result.add_check(
                name="confidence_consistency",
                passed=True,
                detail="Confidence markers are internally consistent.",
                severity=Severity.INFO,
            )

    def _check_hedge_ratio(self, text: str, result: VerificationResult) -> None:
        """Flag excessive hedging (too many uncertainty words)."""
        words = text.lower().split()
        if len(words) < 20:
            result.add_check(
                name="hedge_ratio",
                passed=True,
                detail="Text too short for hedge ratio analysis.",
                severity=Severity.INFO,
            )
            return

        hedge_count = sum(1 for w in words if w.strip(".,;:!?\"'()[]{}") in self.HEDGE_WORDS)
        ratio = hedge_count / len(words)

        if ratio > 0.12:  # >12% hedge words
            result.add_check(
                name="hedge_ratio",
                passed=False,
                detail=(
                    f"Excessive hedging: {hedge_count}/{len(words)} "
                    f"words are uncertainty markers ({ratio:.1%}). "
                    "Consider making stronger, clearer statements."
                ),
                severity=Severity.WARNING,
            )
        else:
            result.add_check(
                name="hedge_ratio",
                passed=True,
                detail=f"Hedge ratio OK: {ratio:.1%}.",
                severity=Severity.INFO,
            )

    def _build_suggestion(self, result: VerificationResult) -> str:
        """Generate a human-readable suggestion from fact-check findings."""
        if result.passed and not result.warnings:
            return ""
        parts: list[str] = []
        w = result.warnings
        e = result.errors
        if e:
            parts.append(f"Factual issues detected: {'; '.join(e)}")
        if w:
            parts.append(f"Potential concerns: {'; '.join(w)}")
        return " ".join(parts)


# ── Unified Verifier ─────────────────────────────────────────────────────────


class Verifier:
    """Unified verification entry point.

    Orchestrates sub-verifiers based on brain type and output content.
    Custom checkers can be registered via `register()`.

    Usage:
        verifier = Verifier()
        result = verifier.verify(response)

        # Register a custom checker
        verifier.register("my_check", lambda r, c: my_result)
    """

    def __init__(self) -> None:
        self._code_verifier = CodeVerifier()
        self._fact_verifier = FactVerifier()
        self._custom_checkers: list[tuple[str, Callable[..., VerificationResult]]] = []

    def register(
        self, name: str, checker: Callable[..., VerificationResult]
    ) -> None:
        """Register a custom verification checker.

        Args:
            name: Unique name for this checker (used in logs).
            checker: Callable that accepts (response, context) and returns
                     a VerificationResult.
        """
        self._custom_checkers.append((name, checker))

    def verify(
        self, response: BrainResponse, context: Optional[Any] = None
    ) -> VerificationResult:
        """Verify a brain response through all applicable checkers.

        Strategy (by brain_type):
            - CODE generation (ACTION brain)  → CodeVerifier + FactVerifier
            - All other brain types            → FactVerifier only

        Args:
            response: The BrainResponse to verify.
            context: Optional additional context (e.g., original request).

        Returns:
            Aggregated VerificationResult.
        """
        master = VerificationResult()

        # ── Code verification (only for code output) ──────────────────
        if self._should_verify_code(response):
            code_result = self._code_verifier.verify(response.content)
            master.merge(code_result)

        # ── Fact verification (always, for all text output) ──────────
        if response.content:
            fact_result = self._fact_verifier.verify(
                response.content, context=context
            )
            master.merge(fact_result)

        # ── Custom checkers ───────────────────────────────────────────
        for name, checker in self._custom_checkers:
            try:
                custom_result = checker(response, context)
                custom_result.checks = [
                    {**c, "checker": name} for c in custom_result.checks
                ]
                master.merge(custom_result)
            except Exception as exc:
                master.add_check(
                    name=f"checker_error:{name}",
                    passed=False,
                    detail=f"Custom checker '{name}' raised: {exc}",
                    severity=Severity.WARNING,
                )

        # ── Final summary ─────────────────────────────────────────────
        master.suggestion = self._build_final_suggestion(master, response)

        return master

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _should_verify_code(response: BrainResponse) -> bool:
        """Determine if this response contains code that should be verified."""
        # Direct code generation by ACTION brain
        if response.brain_type == BrainType.ACTION:
            return True

        # Heuristic: content looks like Python code
        content = response.content.strip()
        if content.startswith("```python") or content.startswith("```py"):
            return True
        if content.startswith("import ") or content.startswith("from "):
            return True
        if content.startswith("def ") or content.startswith("class "):
            return True
        if content.startswith("#!/usr/bin/env python") or content.startswith("#!"):
            return True

        return False

    @staticmethod
    def _build_final_suggestion(
        result: VerificationResult, response: BrainResponse
    ) -> str:
        """Build the final suggestion string."""
        if result.passed and not result.warnings:
            return ""

        parts: list[str] = []

        if not result.passed:
            parts.append(
                f"Verification FAILED ({len(result.errors)} errors). "
                f"Brain: {response.brain_type.value}, "
                f"Confidence: {response.confidence:.0%}"
            )
        elif result.warnings:
            parts.append(
                f"Verification passed with {len(result.warnings)} warning(s). "
                f"Brain: {response.brain_type.value}"
            )

        if result.suggestion:
            parts.append(result.suggestion)

        total_checks = len(result.checks)
        failed_checks = sum(1 for c in result.checks if not c["passed"])
        parts.append(
            f"({failed_checks}/{total_checks} checks flagged)"
        )

        return " | ".join(parts)
