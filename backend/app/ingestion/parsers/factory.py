"""Extensible Parser Factory and Registry adhering to the Open-Closed Principle (OCP)."""

from collections.abc import Callable

from app.ingestion.exceptions import UnsupportedFileTypeError
from app.ingestion.models import DocumentType
from app.ingestion.parsers.base import BaseParser
from app.ingestion.parsers.docx import DocxParser
from app.ingestion.parsers.pdf import PDFParser
from app.ingestion.parsers.text import TextParser


class ParserFactory:
    """Registry and factory for document parsers."""

    _registry: dict[DocumentType, type[BaseParser]] = {}

    @classmethod
    def register(cls, doc_type: DocumentType, parser_cls: type[BaseParser]) -> None:
        """Registers a parser class for a specific DocumentType."""
        cls._registry[doc_type] = parser_cls

    @classmethod
    def get_parser(cls, doc_type: DocumentType) -> BaseParser:
        """Retrieves and instantiates the parser associated with the DocumentType.

        Raises:
            UnsupportedFileTypeError: if no parser has been registered for this type.
        """
        parser_cls = cls._registry.get(doc_type)
        if not parser_cls:
            raise UnsupportedFileTypeError(
                f"No registered parser available for document type: {doc_type.value}"
            )
        return parser_cls()


def register_parser(doc_type: DocumentType) -> Callable[[type[BaseParser]], type[BaseParser]]:
    """Decorator to register a custom parser with the ParserFactory."""

    def decorator(cls: type[BaseParser]) -> type[BaseParser]:
        ParserFactory.register(doc_type, cls)
        return cls

    return decorator


# Register built-in default parsers
ParserFactory.register(DocumentType.PDF, PDFParser)
ParserFactory.register(DocumentType.DOCX, DocxParser)
ParserFactory.register(DocumentType.TXT, TextParser)
ParserFactory.register(DocumentType.MARKDOWN, TextParser)
