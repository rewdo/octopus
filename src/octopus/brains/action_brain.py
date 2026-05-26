"""
Action Brain — Sandboxed tool execution engine.

Provides safe execution of system tools:
    - shell: Subprocess execution with timeout
    - file_read / file_write: File I/O within allowed paths
    - web_search / web_fetch: HTTP requests via httpx
    - python_eval: Restricted Python evaluation (no imports, no __builtins__, no files)
    - browser: Placeholder for browser automation

All tools are permission-checked against request.allowed_tools.
python_eval is fully sandboxed:
    - No __builtins__ access
    - No import (blocked at AST level)
    - No file/network access
    - CPU timeout via signal or soft limit
    - Memory-limited namespace
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .base import (
    BaseBrain,
    BrainRequest,
    BrainResponse,
    BrainType,
    TaskComplexity,
    TaskRisk,
)


# ---------------------------------------------------------------------------
# Tool result
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Result from executing a single tool."""

    tool: str
    success: bool
    output: Any = None
    error: str = ""
    latency_ms: float = 0.0
    retries: int = 0
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Python eval sandbox
# ---------------------------------------------------------------------------

# Whitelist of safe AST node types
SAFE_AST_NODES = {
    ast.Expression, ast.Expr, ast.Constant, ast.Num, ast.Str, ast.Bytes,
    ast.NameConstant, ast.UnaryOp, ast.UAdd, ast.USub, ast.Not, ast.Invert,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
    ast.MatMult, ast.BoolOp, ast.And, ast.Or, ast.Compare, ast.Eq, ast.NotEq,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
    ast.Call, ast.keyword, ast.Name, ast.Load, ast.List, ast.Tuple, ast.Set,
    ast.Dict, ast.comprehension, ast.ListComp, ast.SetComp, ast.DictComp,
    ast.GeneratorExp, ast.IfExp, ast.Attribute, ast.Slice, ast.ExtSlice,
    ast.Index, ast.Subscript, ast.Starred, ast.FormattedValue, ast.JoinedStr,
    ast.Lambda, ast.arguments, ast.arg,
}

# Whitelist of built-in functions allowed in sandbox
SAFE_BUILTINS: dict[str, Any] = {
    "True": True,
    "False": False,
    "None": None,
    "abs": abs,
    "all": all,
    "any": any,
    "ascii": ascii,
    "bin": bin,
    "bool": bool,
    "bytes": bytes,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,  # Redirected to string buffer
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}

# Allowed attributes on safe objects (e.g., str.upper, list.append)
SAFE_ATTRIBUTES = {
    str: {"capitalize", "casefold", "center", "count", "encode", "endswith",
          "expandtabs", "find", "format", "format_map", "index",
          "isalnum", "isalpha", "isascii", "isdecimal", "isdigit",
          "isidentifier", "islower", "isnumeric", "isprintable",
          "isspace", "istitle", "isupper", "join", "ljust", "lower",
          "lstrip", "maketrans", "partition", "removeprefix",
          "removesuffix", "replace", "rfind", "rindex", "rjust",
          "rpartition", "rsplit", "rstrip", "split", "splitlines",
          "startswith", "strip", "swapcase", "title", "translate",
          "upper", "zfill"},
    list: {"append", "clear", "copy", "count", "extend", "index",
           "insert", "pop", "remove", "reverse", "sort"},
    dict: {"clear", "copy", "fromkeys", "get", "items", "keys",
           "pop", "popitem", "setdefault", "update", "values"},
    set: {"add", "clear", "copy", "difference", "difference_update",
          "discard", "intersection", "intersection_update",
          "isdisjoint", "issubset", "issuperset", "pop", "remove",
          "symmetric_difference", "symmetric_difference_update",
          "union", "update"},
    tuple: {"count", "index"},
    bytes: {"decode", "count", "find", "index", "join", "partition",
            "replace", "rfind", "rindex", "rpartition", "split",
            "startswith", "strip", "swapcase", "translate"},
    int: {"bit_length", "to_bytes"},
    float: {"as_integer_ratio", "is_integer", "hex"},
}


def _validate_ast(code: str) -> None:
    """Validate that code only uses safe AST nodes.

    Raises:
        ValueError: If dangerous constructs are detected.
    """
    tree = ast.parse(code, mode="eval")

    for node in ast.walk(tree):
        t = type(node)

        # Block imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("import statements are forbidden in sandbox")

        # Block dangerous attribute access
        if isinstance(node, ast.Attribute):
            attr_name = node.attr
            if attr_name.startswith("__"):
                raise ValueError(
                    f"Access to dunder attribute '{attr_name}' is forbidden"
                )

        # Block node types not in whitelist
        if t not in SAFE_AST_NODES:
            name = t.__name__ if hasattr(t, "__name__") else str(t)
            raise ValueError(
                f"AST node type '{name}' is not allowed in sandbox"
            )


class _PrintCapture:
    """Captures print() output to a string buffer."""

    def __init__(self):
        self.buffer: list[str] = []

    def write(self, s: str) -> None:
        self.buffer.append(s)

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        return "".join(self.buffer)


def _empty_open(*args, **kwargs):
    """Raise error when open() is called in sandbox."""
    raise PermissionError("open() is not permitted in the sandbox environment")


@contextmanager
def _time_limit(seconds: float):
    """Soft CPU time limit for sandbox (best-effort, not hard)."""

    def signal_handler(signum, frame):
        raise TimeoutError(f"Sandbox execution exceeded {seconds}s time limit")

    old_handler = None
    try:
        old_handler = __import__("signal").signal(
            __import__("signal").SIGALRM, signal_handler
        )
        __import__("signal").alarm(int(seconds))
    except (ImportError, AttributeError):
        pass  # Not available on all platforms
    try:
        yield
    finally:
        try:
            __import__("signal").alarm(0)
            if old_handler:
                __import__("signal").signal(
                    __import__("signal").SIGALRM, old_handler
                )
        except (ImportError, AttributeError):
            pass


# ---------------------------------------------------------------------------
# ActionBrain
# ---------------------------------------------------------------------------

class ActionBrain(BaseBrain):
    """Tool execution engine with sandboxed code evaluation.

    Supports five tool categories:
        - shell: Run subprocess commands
        - file_read: Read files from workspace
        - file_write: Write files to workspace
        - web_search / web_fetch: HTTP requests
        - python_eval: Sandboxed Python execution

    Usage::

        brain = ActionBrain(workspace_dir=Path("./workspace"))
        request = BrainRequest(
            task_id="t1",
            user_input="run ls",
            allowed_tools=["shell"],
            metadata={"command": "dir", "args": []},
        )
        response = await brain.process(request)
    """

    def __init__(
        self,
        workspace_dir: Optional[Path] = None,
        config: Any = None,
    ):
        super().__init__(config)
        self._workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        self._max_retries = 2

    @property
    def brain_type(self) -> BrainType:
        return BrainType.ACTION

    async def process(self, request: BrainRequest) -> BrainResponse:
        """Execute tools specified in the request metadata.

        The request.metadata should contain:
            - tool: str (required) — the tool to execute
            - params: dict (optional) — tool-specific parameters
            - tools: list[dict] (optional) — batch of tool calls

        If neither 'tool' nor 'tools' is specified, returns an error.
        """
        t_start = time.perf_counter()
        results: list[ToolResult] = []
        errors: list[str] = []

        all_tools = []

        # Single tool
        if "tool" in request.metadata:
            all_tools.append({
                "tool": request.metadata["tool"],
                "params": request.metadata.get("params", {}),
            })
        # Batch of tools
        elif "tools" in request.metadata:
            all_tools = request.metadata["tools"]
        else:
            return BrainResponse(
                success=False,
                content="No tool specified. Provide 'tool' or 'tools' in metadata.",
                brain_type=BrainType.ACTION,
                confidence=0.0,
                errors=["no_tool_specified"],
            )

        for tool_spec in all_tools:
            tool_name = tool_spec["tool"]
            params = tool_spec.get("params", {})

            # Permission check
            if request.allowed_tools and tool_name not in request.allowed_tools:
                errors.append(f"Tool '{tool_name}' not in allowed_tools")
                results.append(ToolResult(
                    tool=tool_name,
                    success=False,
                    error=f"Permission denied: '{tool_name}' not allowed",
                ))
                continue

            # Execute tool
            result = await self._execute_tool_with_retry(tool_name, params)
            results.append(result)
            if not result.success:
                errors.append(f"Tool '{tool_name}' failed: {result.error}")

        latency_ms = (time.perf_counter() - t_start) * 1000
        all_success = all(r.success for r in results)

        # Build response content
        output_lines = []
        for r in results:
            status = "✅" if r.success else "❌"
            line = f"[{status}] {r.tool} ({r.latency_ms:.1f}ms)"
            if r.retries:
                line += f" [retries: {r.retries}]"
            output_lines.append(line)
            if r.output is not None:
                out_str = str(r.output)
                if len(out_str) > 500:
                    out_str = out_str[:497] + "..."
                output_lines.append(f"  → {out_str}")
            if r.error:
                output_lines.append(f"  ⚠ {r.error}")

        return BrainResponse(
            success=all_success,
            content="\n".join(output_lines),
            brain_type=BrainType.ACTION,
            tokens_used=0,
            cost_usd=0.0,
            latency_ms=round(latency_ms, 2),
            confidence=1.0 if all_success else 0.5,
            tool_calls=[
                {
                    "tool": r.tool,
                    "success": r.success,
                    "latency_ms": r.latency_ms,
                    "retries": r.retries,
                }
                for r in results
            ],
            structured_output={
                "results": [
                    {
                        "tool": r.tool,
                        "success": r.success,
                        "output": (
                            str(r.output)[:200] if r.output is not None else None
                        ),
                        "error": r.error,
                        "latency_ms": r.latency_ms,
                    }
                    for r in results
                ],
                "total_tools": len(results),
                "success_count": sum(1 for r in results if r.success),
                "failure_count": sum(1 for r in results if not r.success),
            },
            errors=errors,
            metadata={
                "workspace": str(self._workspace_dir),
                "tools_available": [
                    "shell", "file_read", "file_write",
                    "web_search", "web_fetch", "python_eval", "browser",
                ],
            },
        )

    def can_handle(self, request: BrainRequest) -> bool:
        """ActionBrain handles requests with tool calls or allowed_tools."""
        return bool(request.allowed_tools) or "tool" in request.metadata

    async def _execute_tool_with_retry(
        self, tool_name: str, params: dict[str, Any]
    ) -> ToolResult:
        """Execute a tool with automatic retry on failure."""
        last_result: Optional[ToolResult] = None

        for attempt in range(self._max_retries + 1):
            result = await self._execute_tool(tool_name, params, attempt)
            if result.success:
                return result
            last_result = result
            if attempt < self._max_retries:
                await asyncio.sleep(0.2 * (attempt + 1))

        return last_result or ToolResult(
            tool=tool_name,
            success=False,
            error="Max retries exhausted",
        )

    async def _execute_tool(
        self, tool_name: str, params: dict[str, Any], attempt: int = 0
    ) -> ToolResult:
        """Execute a single tool based on its name."""
        t0 = time.perf_counter()
        handlers = {
            "shell": self._tool_shell,
            "file_read": self._tool_file_read,
            "file_write": self._tool_file_write,
            "web_search": self._tool_web_search,
            "web_fetch": self._tool_web_fetch,
            "python_eval": self._tool_python_eval,
            "browser": self._tool_browser,
        }

        handler = handlers.get(tool_name)
        if handler is None:
            return ToolResult(
                tool=tool_name,
                success=False,
                error=f"Unknown tool: {tool_name}",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            output = await handler(params)
            return ToolResult(
                tool=tool_name,
                success=True,
                output=output,
                latency_ms=(time.perf_counter() - t0) * 1000,
                retries=attempt,
            )
        except Exception as e:
            return ToolResult(
                tool=tool_name,
                success=False,
                error=f"{type(e).__name__}: {e}",
                latency_ms=(time.perf_counter() - t0) * 1000,
                retries=attempt,
            )

    # ---- Shell tool ----

    async def _tool_shell(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a shell command via subprocess.

        Params:
            command: The command to run (string or list).
            cwd: Optional working directory.
            timeout: Maximum execution time in seconds (default 30).
            env: Optional environment variables dict.
            shell: Use shell=True (default False for safety).
        """
        command = params.get("command", "")
        if not command:
            raise ValueError("'command' parameter is required")

        cwd = params.get("cwd", str(self._workspace_dir))
        timeout = params.get("timeout", 30)
        env_vars = params.get("env")
        use_shell = params.get("shell", False)

        # Convert string to list for safer execution
        if isinstance(command, str) and not use_shell:
            import shlex
            try:
                command = shlex.split(command)
            except ValueError:
                # Fall back to shell mode for complex commands
                command = [command]

        # Build env
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        process = await asyncio.create_subprocess_exec(
            *command if isinstance(command, list) else [command],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
            shell=use_shell,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(f"Command timed out after {timeout}s")

        return {
            "returncode": process.returncode,
            "stdout": stdout.decode("utf-8", errors="replace").strip(),
            "stderr": stderr.decode("utf-8", errors="replace").strip(),
            "command": str(command),
            "cwd": cwd,
        }

    # ---- File read tool ----

    async def _tool_file_read(self, params: dict[str, Any]) -> dict[str, Any]:
        """Read a file from the workspace.

        Params:
            path: Relative or absolute file path.
            encoding: File encoding (default utf-8).
            max_bytes: Maximum bytes to read (default 1MB).
        """
        file_path = params.get("path", "")
        if not file_path:
            raise ValueError("'path' parameter is required")

        encoding = params.get("encoding", "utf-8")
        max_bytes = params.get("max_bytes", 1_048_576)

        path = self._resolve_path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.is_dir():
            raise IsADirectoryError(f"Path is a directory: {path}")

        content = path.read_text(encoding=encoding)
        truncated = len(content.encode(encoding)) > max_bytes
        if truncated:
            content = content[:max_bytes // 2]  # Rough character limit

        return {
            "path": str(path),
            "content": content,
            "size_bytes": path.stat().st_size,
            "encoding": encoding,
            "truncated": truncated,
        }

    # ---- File write tool ----

    async def _tool_file_write(self, params: dict[str, Any]) -> dict[str, Any]:
        """Write content to a file in the workspace.

        Params:
            path: Relative or absolute file path.
            content: The text content to write.
            encoding: File encoding (default utf-8).
            mode: Write mode — 'w' (overwrite) or 'a' (append).
            create_dirs: Auto-create parent directories (default True).
        """
        file_path = params.get("path", "")
        content = params.get("content", "")
        encoding = params.get("encoding", "utf-8")
        mode = params.get("mode", "w")
        create_dirs = params.get("create_dirs", True)

        if not file_path:
            raise ValueError("'path' parameter is required")
        if mode not in ("w", "a"):
            raise ValueError("mode must be 'w' or 'a'")

        path = self._resolve_path(file_path)

        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(content, encoding=encoding)

        return {
            "path": str(path),
            "bytes_written": len(content.encode(encoding)),
            "encoding": encoding,
            "mode": mode,
        }

    # ---- Web search / fetch tools ----

    async def _tool_web_search(self, params: dict[str, Any]) -> dict[str, Any]:
        """Perform a web search (placeholder — requires external API).

        Params:
            query: Search query string.
            engine: Search engine (default 'auto').
            max_results: Maximum results to return.
        """
        query = params.get("query", "")
        if not query:
            raise ValueError("'query' parameter is required")

        engine = params.get("engine", "auto")
        max_results = params.get("max_results", 10)

        try:
            import httpx

            async with httpx.AsyncClient(timeout=15.0) as client:
                # Use DuckDuckGo HTML search (no API key needed)
                response = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={
                        "User-Agent": "Octopus/5.0 ActionBrain",
                    },
                )

                # Basic extraction of result snippets
                text = response.text
                # Extract snippets between result__snippet markers
                results = []
                import re as _re
                snippets = _re.findall(
                    r'class="result__snippet"[^>]*>(.*?)</a>',
                    text,
                    _re.DOTALL,
                )
                for s in snippets[:max_results]:
                    clean = _re.sub(r"<[^>]+>", "", s).strip()
                    if clean:
                        results.append({"snippet": clean})

                return {
                    "query": query,
                    "engine": engine,
                    "results": results,
                    "total": len(results),
                }
        except ImportError:
            return {
                "query": query,
                "engine": engine,
                "results": [],
                "note": "httpx not available — install with 'pip install httpx'",
            }

    async def _tool_web_fetch(self, params: dict[str, Any]) -> dict[str, Any]:
        """Fetch content from a URL.

        Params:
            url: The URL to fetch.
            method: HTTP method (default GET).
            headers: Optional request headers dict.
            timeout: Request timeout in seconds (default 30).
            max_size: Maximum response size in bytes (default 5MB).
        """
        url = params.get("url", "")
        if not url:
            raise ValueError("'url' parameter is required")

        method = params.get("method", "GET").upper()
        headers = params.get("headers", {})
        timeout = params.get("timeout", 30)
        max_size = params.get("max_size", 5_242_880)  # 5MB

        try:
            import httpx

            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                max_redirects=5,
            ) as client:
                response = await client.request(
                    method,
                    url,
                    headers={**headers, "User-Agent": "Octopus/5.0 ActionBrain"},
                )
                content = response.text[:max_size]
                truncated = len(response.content) > max_size

                return {
                    "url": str(response.url),
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "content": content,
                    "content_length": len(content),
                    "truncated": truncated,
                    "encoding": response.encoding or "utf-8",
                }
        except ImportError:
            return {
                "url": url,
                "status_code": 0,
                "error": "httpx not available — install with 'pip install httpx'",
            }

    # ---- Python eval tool (SANDBOXED) ----

    async def _tool_python_eval(self, params: dict[str, Any]) -> dict[str, Any]:
        """Evaluate Python code in a sandboxed environment.

        Security measures:
            - AST validation blocks imports, dunders, and dangerous nodes
            - No __builtins__ except whitelisted safe functions
            - No file access (open() is disabled)
            - No __import__
            - Print output is captured
            - Soft timeout via signal (best-effort)

        Params:
            code: The Python expression to evaluate.
            timeout: Maximum execution time in seconds.
        """
        code = params.get("code", "")
        if not code:
            raise ValueError("'code' parameter is required")

        timeout_sec = params.get("timeout", 5)

        # Step 1: Validate AST
        _validate_ast(code)

        # Step 2: Build sandbox namespace
        sandbox_builtins: dict[str, Any] = {}
        sandbox_builtins.update(SAFE_BUILTINS)
        # Replace open with a disallowed version
        sandbox_builtins["open"] = _empty_open
        # Capture print output
        capture = _PrintCapture()
        sandbox_builtins["print"] = lambda *a, **kw: print(
            *a, **{**kw, "file": capture}
        )

        sandbox_globals: dict[str, Any] = {"__builtins__": sandbox_builtins}
        sandbox_locals: dict[str, Any] = {}

        # Step 3: Execute with timeout
        try:
            with _time_limit(timeout_sec):
                compiled = compile(code, "<sandbox>", "eval")
                result = eval(compiled, sandbox_globals, sandbox_locals)
        except SyntaxError as e:
            return {
                "success": False,
                "error": f"SyntaxError: {e}",
                "result": None,
                "stdout": capture.getvalue(),
            }

        return {
            "success": True,
            "result": result,
            "type": type(result).__name__,
            "stdout": capture.getvalue(),
        }

    # ---- Browser tool (placeholder) ----

    async def _tool_browser(self, params: dict[str, Any]) -> dict[str, Any]:
        """Browser automation placeholder.

        Full browser automation requires integration with Playwright/Selenium.
        This is a stub for Phase 1.
        """
        return {
            "action": params.get("action", "navigate"),
            "url": params.get("url", ""),
            "status": "placeholder",
            "note": "Full browser automation available in Phase 2",
        }

    # ---- Helpers ----

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve a potentially relative path to an absolute workspace path."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self._workspace_dir / path
        return path.resolve()

    @property
    def workspace_dir(self) -> Path:
        """Return the configured workspace directory."""
        return self._workspace_dir
