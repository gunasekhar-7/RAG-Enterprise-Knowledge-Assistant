import re

import structlog

logger = structlog.get_logger(__name__)


class SecurityException(Exception):
    """Raised when a security guardrail is violated."""

    pass


class InputValidator:
    """
    Validates user input queries against known prompt injection heuristics.
    """

    # Common prompt injection signatures and bypass attempts
    INJECTION_PATTERNS = [
        r"(?i)\bignore\b.*\b(previous|above|all)\b.*\b(instructions|directions|rules)\b",
        r"(?i)\bforget\b.*\b(previous|above|all)\b.*\b(instructions|directions|rules)\b",
        r"(?i)\bdisregard\b.*\b(previous|above|all)\b.*\b(instructions|directions|rules)\b",
        r"(?i)\bsystem prompt\b",
        r"(?i)\bprint\b.*\b(instructions|rules|prompt)\b",
        r"(?i)\bbypass\b",
        r"(?i)\byou are now\b",
        r"(?i)\bact as\b.*\b(unrestricted|jailbreak|DAN)\b",
    ]

    def __init__(self):  # type: ignore
        self.compiled_patterns = [re.compile(p) for p in self.INJECTION_PATTERNS]

    def validate(self, query: str) -> bool:
        """
        Validates the query. Raises SecurityException if malicious.
        Returns True if safe.
        """
        if not query or not query.strip():
            return True

        for pattern in self.compiled_patterns:
            if pattern.search(query):
                logger.warning("Prompt injection detected", query=query, pattern=pattern.pattern)
                raise SecurityException("Query blocked by security guardrails.")

        return True
