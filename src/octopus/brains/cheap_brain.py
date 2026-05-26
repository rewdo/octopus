"""
Cheap Brain — Ultra-low-cost intent classification and entity extraction.

Phase 1: Fully rule-based with zero external dependencies.
Performance target: < 10ms per request.

Responsibilities:
    - Intent classification via keyword + regex rules
    - Entity extraction (dates, emails, URLs, numbers, names)
    - Simple FAQ matching
    - Confidence scoring for routing decisions
"""

from __future__ import annotations

import re
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
# Intent definitions and keyword patterns
# ---------------------------------------------------------------------------

INTENT_PATTERNS: dict[str, dict[str, Any]] = {
    "greeting": {
        "keywords": [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "greetings", "howdy", "yo", "sup",
            "你好", "您好", "嗨", "早上好", "下午好", "晚上好",
            "好久不见", "在吗", "在不在",
        ],
        "responses": [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Hey! Ready to help. What do you need?",
        ],
        "complexity": TaskComplexity.TRIVIAL,
    },
    "farewell": {
        "keywords": [
            "bye", "goodbye", "see you", "later", "farewell",
            "再见", "拜拜", "回头见", "下次见",
        ],
        "responses": [
            "Goodbye! Let me know if you need anything else.",
            "See you later! Take care.",
            "Bye! Happy to help anytime.",
        ],
        "complexity": TaskComplexity.TRIVIAL,
    },
    "question": {
        "keywords": [
            "what", "why", "how", "when", "where", "who",
            "which", "can you", "could you", "do you",
            "什么是", "为什么", "怎么", "如何", "谁", "哪里",
            "什么时候", "能不能", "可以", "吗？", "吗?",
        ],
        "patterns": [
            r"^(what|why|how|when|where|who|which|can|could|do|does|is|are|was|were|will|would|should|shall|did|has|have)\b.*\?",
            r".*吗[？?]$",
            r".*[？?]$",
        ],
        "complexity": TaskComplexity.SIMPLE,
    },
    "command": {
        "keywords": [
            "create", "delete", "update", "change", "set", "get",
            "run", "start", "stop", "build", "install", "remove",
            "add", "edit", "open", "close", "save", "load",
            "list", "show", "display", "find", "search", "move",
            "copy", "rename", "convert", "merge", "split",
            "创建", "删除", "更新", "修改", "设置", "获取",
            "运行", "启动", "停止", "构建", "安装", "卸载",
            "添加", "编辑", "打开", "关闭", "保存", "加载",
            "列出", "显示", "查找", "搜索", "移动",
            "复制", "重命名", "转换", "合并", "拆分",
        ],
        "complexity": TaskComplexity.MODERATE,
    },
    "code_related": {
        "keywords": [
            "code", "function", "class", "bug", "debug", "error",
            "import", "def ", "return", "variable", "loop",
            "algorithm", "api", "endpoint", "database", "query",
            "python", "javascript", "typescript", "rust", "go",
            "docker", "kubernetes", "aws", "server", "client",
            "refactor", "optimize", "compile", "deploy", "test",
            "代码", "函数", "类", "错误", "调试",
            "导入", "变量", "循环", "算法", "接口",
            "数据库", "查询", "重构", "优化", "编译",
            "部署", "测试",
        ],
        "complexity": TaskComplexity.COMPLEX,
    },
    "translation": {
        "keywords": [
            "translate", "translation", "tr to",
            "翻译", "译成", "翻成", "转为",
        ],
        "patterns": [
            r"\b(translate|tr(anslate)?)\b.*\b(to|into|from)\b",
            r"\bin\s+(\w+)\b.*\b(in\s+)?(\w+)\b",
            r"翻译[成为到]",
        ],
        "complexity": TaskComplexity.SIMPLE,
    },
    "summarization": {
        "keywords": [
            "summarize", "summary", "tldr", "tl;dr", "recap",
            "brief", "condense", "shorten", "bullet points",
            "总结", "概括", "摘要", "归纳", "精简",
        ],
        "patterns": [
            r"\b(summarize|summary|tldr|recap)\b",
            r"(in\s+a\s+)?(summary|summarize)",
        ],
        "complexity": TaskComplexity.MODERATE,
    },
    "calculation": {
        "keywords": [
            "calculate", "compute", "solve", "equation",
            "math", "arithmetic", "add", "subtract",
            "multiply", "divide", "sum", "average", "total",
            "计算", "算式", "数学", "加减乘除",
            "求和", "平均", "总数",
        ],
        "patterns": [
            r"^\s*[\d\s\+\-\*\/\(\)\.\^\%]+=?$",
            r"\b(\d+)\s*[\+\-\*\/]\s*(\d+)\b",
        ],
        "complexity": TaskComplexity.SIMPLE,
    },
    "file_operation": {
        "keywords": [
            "file", "folder", "directory", "path",
            "read", "write", "append", "save",
            "open file", "file content", "text file",
            "文件", "目录", "文件夹", "路径",
            "读取", "写入", "保存文件",
        ],
        "complexity": TaskComplexity.MODERATE,
    },
    "web_search": {
        "keywords": [
            "search", "google", "look up", "find online",
            "internet", "web", "online", "browse",
            "搜索", "查一下", "上网查", "网上",
        ],
        "complexity": TaskComplexity.MODERATE,
    },
    "self_identity": {
        "keywords": [
            "who are you", "what are you", "your name",
            "introduce yourself", "help me understand you",
            "你是谁", "你叫什么", "介绍一下",
            "你是什么", "你的名字", "介绍一下自己",
        ],
        "responses": [
            "I'm Octopus, a multi-brain AI agent. I use different specialized brains to handle tasks efficiently — from simple classification to complex reasoning. How can I help you?",
        ],
        "complexity": TaskComplexity.TRIVIAL,
    },
}


# ---------------------------------------------------------------------------
# Entity extraction patterns
# ---------------------------------------------------------------------------

ENTITY_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    "url": re.compile(
        r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?::\d+)?(?:/[-\w%!$&'()*+,;=.:@?/~#]*)?"
    ),
    "phone_cn": re.compile(
        r"(?:(?:\+?86)|(?:\(\+86\)))?\s*1[3-9]\d{9}"
    ),
    "phone_us": re.compile(
        r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    ),
    "date_iso": re.compile(
        r"\b\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b"
    ),
    "date_cn": re.compile(
        r"\b\d{4}年(?:0?[1-9]|1[0-2])月(?:0?[1-9]|[12]\d|3[01])[日号]?\b"
    ),
    "date_us": re.compile(
        r"\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/\d{2,4}\b"
    ),
    "time_24h": re.compile(
        r"\b(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\b"
    ),
    "ip_v4": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
    "file_path_win": re.compile(
        r"\b[A-Za-z]:\\(?:[^\s\\/:*?\"<>|]+\\)*[^\s\\/:*?\"<>|]*"
    ),
    "file_path_unix": re.compile(
        r"(?:/~[^\s]*|/[^\s]*(?:\.\w+))\b"
    ),
    "chinese_name": re.compile(
        r"[\u4e00-\u9fff]{2,4}(?:先生|女士|老师|同志)?"
    ),
    "number_int": re.compile(
        r"(?<!\w)-?\b\d+\b(?!\.\d)"
    ),
    "number_float": re.compile(
        r"-?\b\d+\.\d+\b"
    ),
    "percentage": re.compile(
        r"\b\d+(?:\.\d+)?\s*%"
    ),
    "currency": re.compile(
        r"[¥$€£]\s*\d+(?:,\d{3})*(?:\.\d{2})?"
    ),
    "hashtag": re.compile(
        r"#\w+"
    ),
    "mention": re.compile(
        r"@\w+"
    ),
    "uuid": re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    ),
    "md5": re.compile(
        r"\b[a-fA-F0-9]{32}\b"
    ),
}


# ---------------------------------------------------------------------------
# FAQ rules (simple question → answer mapping)
# ---------------------------------------------------------------------------

FAQ_RULES: dict[str, list[str]] = {
    "what is octopus": [
        "Octopus is a multi-brain AI agent that uses specialized brains "
        "(Cheap, Skill, Action, Planning, Memory, World, Frontier) to "
        "route tasks to the most cost-effective and capable brain."
    ],
    "what brain should i use": [
        "It depends on your task. The router automatically selects: "
        "Cheap Brain for simple queries, Skill Brain for known workflows, "
        "Action Brain for tool use, and Frontier Brain for complex reasoning."
    ],
    "how does octopus work": [
        "Octopus uses a Cognitive Router to score tasks across 7 dimensions "
        "(complexity, risk, novelty, etc.) and routes them to one of seven "
        "specialized brains for optimal cost-performance balance."
    ],
    "what is cheap brain": [
        "Cheap Brain is the lowest-cost brain. It uses pure rule-based "
        "matching (no API calls) for intent classification, entity extraction, "
        "and simple Q&A. Latency < 10ms, cost = $0."
    ],
    "what languages do you support": [
        "I support English and Chinese text processing, with plans to "
        "add more languages in future versions."
    ],
    "hello": [
        "Hello! How can I help you today?",
    ],
    "hi": [
        "Hi there! What can I do for you?",
    ],
    "你好": [
        "你好！有什么可以帮你的？",
    ],
    "谢谢": [
        "不客气！随时为你效劳。",
        "不用谢，这是我应该做的。",
    ],
    "thank you": [
        "You're welcome! Happy to help.",
        "No problem! Let me know if you need anything else.",
    ],
    "thanks": [
        "You're welcome!",
        "Glad I could help!",
    ],
    "how are you": [
        "I'm running smoothly, thanks for asking! How can I assist you today?",
        "All systems operational. What can I do for you?",
    ],
    "good night": [
        "Good night! Rest well.",
        "Good night! Talk to you later.",
    ],
}


# ---------------------------------------------------------------------------
# CheapBrain implementation
# ---------------------------------------------------------------------------

class CheapBrain(BaseBrain):
    """Ultra-low-cost brain: intent classification + entities + FAQ — all local.

    Zero external dependencies. All processing is pure Python regex + keyword
    matching. Designed for <10ms response time on commodity hardware.

    Usage::

        brain = CheapBrain()
        request = BrainRequest(task_id="t1", user_input="translate hello to chinese")
        response = await brain.process(request)
        print(response.content, response.confidence)
    """

    @property
    def brain_type(self) -> BrainType:
        return BrainType.CHEAP

    async def process(self, request: BrainRequest) -> BrainResponse:
        """Process a user request through the Cheap Brain pipeline.

        Pipeline:
            1. Intent classification
            2. Entity extraction
            3. FAQ lookup
            4. Compile structured response
        """
        text = request.user_input.strip()
        if not text:
            return BrainResponse(
                success=False,
                content="Empty input.",
                brain_type=BrainType.CHEAP,
                confidence=0.0,
                errors=["empty_input"],
            )

        # Step 1: Intent classification
        intent, intent_conf = self._classify_intent(text)

        # Step 2: Entity extraction
        entities = self._extract_entities(text)

        # Step 3: FAQ lookup
        faq_answer = self._lookup_faq(text)

        # Step 4: Build response
        return self._build_response(
            text=text,
            intent=intent,
            intent_conf=intent_conf,
            entities=entities,
            faq_answer=faq_answer,
        )

    def can_handle(self, request: BrainRequest) -> bool:
        """Cheap Brain can handle any request — it's the default fallback."""
        return True

    # ---- Intent classification ----

    def _classify_intent(self, text: str) -> tuple[str, float]:
        """Classify the user intent using keyword + regex matching.

        Returns:
            Tuple of (intent_name, confidence_score).
            confidence ranges from 0.0 (no match) to 1.0 (perfect match).
        """
        text_lower = text.lower()
        scores: dict[str, float] = {}

        for intent_name, config in INTENT_PATTERNS.items():
            score = 0.0
            keywords = config.get("keywords", [])
            patterns = config.get("patterns", [])

            # Keyword matching
            if keywords:
                hits = sum(1 for kw in keywords if kw.lower() in text_lower)
                # Normalize: cap at 3 hits, then scale
                keyword_score = min(hits / max(len(text.split()) * 0.3, 1), 1.0)
                score += keyword_score * 0.6

            # Regex pattern matching
            if patterns:
                pattern_hits = 0
                for p in patterns:
                    try:
                        if re.search(p, text, re.IGNORECASE):
                            pattern_hits += 1
                    except re.error:
                        continue
                pattern_score = min(pattern_hits / max(len(patterns) * 0.5, 1), 1.0)
                score += pattern_score * 0.4

            if score > 0:
                scores[intent_name] = min(score, 1.0)

        if not scores:
            return "unknown", 0.0

        # Return the highest-scoring intent
        best_intent = max(scores, key=lambda k: scores[k])
        return best_intent, round(scores[best_intent], 3)

    # ---- Entity extraction ----

    def _extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract named entities from text using regex patterns.

        Returns:
            Dict mapping entity type to list of matched strings.
        """
        entities: dict[str, list[str]] = {}

        for entity_type, pattern in ENTITY_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                # Deduplicate while preserving order
                seen: set[str] = set()
                unique: list[str] = []
                for m in matches:
                    if isinstance(m, tuple):
                        m = "".join(m)
                    m = m.strip()
                    if m and m not in seen:
                        seen.add(m)
                        unique.append(m)
                if unique:
                    entities[entity_type] = unique

        return entities

    # ---- FAQ lookup ----

    def _lookup_faq(self, text: str) -> Optional[str]:
        """Try to match the input against the FAQ knowledge base.

        Uses normalized keyword matching for fuzzy matching.
        """
        text_lower = text.lower().strip().rstrip("?.!。？！")

        # Exact match first
        if text_lower in FAQ_RULES:
            import random
            answers = FAQ_RULES[text_lower]
            return random.choice(answers)

        # Partial match: check if any FAQ key is a substring
        for key, answers in FAQ_RULES.items():
            if key in text_lower or text_lower in key:
                import random
                return random.choice(answers)

        return None

    # ---- Response building ----

    def _build_response(
        self,
        text: str,
        intent: str,
        intent_conf: float,
        entities: dict[str, list[str]],
        faq_answer: Optional[str],
    ) -> BrainResponse:
        """Build the final BrainResponse with all extracted information."""
        intent_config = INTENT_PATTERNS.get(intent, {})
        complexity = intent_config.get("complexity", TaskComplexity.SIMPLE)

        # Determine if this can be fully handled or needs escalation
        needs_escalation = faq_answer is None and intent in (
            "command", "code_related", "summarization", "web_search",
        )

        if faq_answer:
            content = faq_answer
            success = True
            confidence = 1.0
        elif intent == "greeting":
            import random
            content = random.choice(intent_config.get("responses", ["Hello!"]))
            success = True
            confidence = 0.95
        elif intent == "farewell":
            import random
            content = random.choice(intent_config.get("responses", ["Goodbye!"]))
            success = True
            confidence = 0.95
        elif intent == "self_identity":
            import random
            content = random.choice(intent_config.get("responses", ["I'm Octopus."]))
            success = True
            confidence = 0.95
        else:
            # Build a diagnostic response for routing
            parts = [f"[Cheap Brain] Intent: {intent} (confidence: {intent_conf:.2f})"]

            if entities:
                entity_str = "; ".join(
                    f"{k}: {', '.join(v)}" for k, v in entities.items()
                )
                parts.append(f"Entities: {entity_str}")

            if needs_escalation:
                parts.append("⚠ Needs escalation to a more capable brain.")
            else:
                parts.append("Ready for further processing.")

            content = "\n".join(parts)
            success = not needs_escalation
            confidence = intent_conf

        # Estimate complexity and risk
        suggested_next: Optional[BrainType] = None
        if needs_escalation:
            if intent in ("command", "file_operation"):
                suggested_next = BrainType.ACTION
            elif intent in ("code_related", "summarization"):
                suggested_next = BrainType.SKILL
            elif intent == "web_search":
                suggested_next = BrainType.PLANNING

        # Risk assessment
        risk = TaskRisk.NONE
        if intent == "command":
            risk = TaskRisk.LOW
        elif intent in ("code_related", "file_operation"):
            risk = TaskRisk.MEDIUM

        return BrainResponse(
            success=success,
            content=content,
            brain_type=BrainType.CHEAP,
            tokens_used=0,
            cost_usd=0.0,
            latency_ms=0.0,  # Measured externally
            confidence=confidence,
            suggested_next_brain=suggested_next,
            should_escalate=needs_escalation,
            escalation_reason=(
                f"Intent '{intent}' requires capabilities beyond Cheap Brain"
                if needs_escalation
                else ""
            ),
            structured_output={
                "intent": intent,
                "intent_confidence": intent_conf,
                "entities": entities,
                "complexity": complexity.name,
                "faq_matched": faq_answer is not None,
            },
            metadata={
                "brain_version": "1.0.0",
                "intent_config_keys": list(INTENT_PATTERNS.keys()),
                "entity_types_found": list(entities.keys()),
            },
        )

    # ---- Public helper methods ----

    def classify(self, text: str) -> tuple[str, float]:
        """Public: classify intent only."""
        return self._classify_intent(text)

    def extract(self, text: str) -> dict[str, list[str]]:
        """Public: extract entities only."""
        return self._extract_entities(text)

    @staticmethod
    def list_intents() -> list[str]:
        """List all supported intent categories."""
        return list(INTENT_PATTERNS.keys())

    @staticmethod
    def list_entity_types() -> list[str]:
        """List all supported entity types."""
        return list(ENTITY_PATTERNS.keys())
