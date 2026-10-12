from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """
    Abstract Base Class for embedding models.
    Provides standard interface for embedding single texts and batches of texts.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the canonical name of the embedding model being used."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the generated vectors."""
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: The vector representation.
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of text strings.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: A list of vector representations corresponding to the input texts.
        """
        pass

    async def initialize(self) -> None:
        """Optional initialization for models that need async setup."""
        pass
