import re

import structlog

logger = structlog.get_logger(__name__)


class PIIScrubber:
    """
    Scrub common Personally Identifiable Information (PII) from strings.
    """

    # Basic patterns for SSN, Credit Card, Phone (US)
    SSN_PATTERN = re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b")
    CC_PATTERN = re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})\b"
    )
    # Extremely basic email pattern
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

    def scrub(self, text: str) -> str:
        """
        Replace matched PII with <REDACTED>.
        """
        if not text:
            return text

        original_text = text
        text = self.SSN_PATTERN.sub("<REDACTED_SSN>", text)
        text = self.CC_PATTERN.sub("<REDACTED_CC>", text)
        text = self.EMAIL_PATTERN.sub("<REDACTED_EMAIL>", text)

        if text != original_text:
            logger.info("PII scrubbed from text")

        return text
