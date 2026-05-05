"""
Abstract base classes for embedding models.

To evaluate your model on RTEB, inherit from one of:
- SentenceTransformerModel: for sentence-transformers compatible models
- OpenSourceModel: for other open-source models loaded from HuggingFace
- APIModel: for proprietary models behind an API

At minimum, you must implement embed_query() and embed_document().
Optionally, implement embed_queries() and embed_documents() for batch processing.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseModel(ABC):
    """Abstract base class for all RTEB embedding models."""

    def __init__(self, model_name: str, distance_metric: str = "cosine"):
        """
        Args:
            model_name: Identifier for this model (used in results)
            distance_metric: Distance metric to use ('cosine' or 'dot')
        """
        self.model_name = model_name
        self.distance_metric = distance_metric

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.

        Args:
            query: The query text

        Returns:
            List of floats representing the query embedding
        """
        pass

    @abstractmethod
    def embed_document(self, document: str) -> List[float]:
        """
        Embed a single document string.

        Args:
            document: The document text

        Returns:
            List of floats representing the document embedding
        """
        pass

    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        """
        Embed a batch of queries. Override for efficient batch processing.

        Args:
            queries: List of query texts

        Returns:
            List of embeddings
        """
        return [self.embed_query(q) for q in queries]

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embed a batch of documents. Override for efficient batch processing.

        Args:
            documents: List of document texts

        Returns:
            List of embeddings
        """
        return [self.embed_document(d) for d in documents]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"


class SentenceTransformerModel(BaseModel):
    """
    Wrapper for sentence-transformers models.

    Usage:
        model = SentenceTransformerModel("sentence-transformers/all-MiniLM-L6-v2")
    """

    def __init__(
        self,
        model_name_or_path: str,
        distance_metric: str = "cosine",
        device: Optional[str] = None,
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        super().__init__(model_name_or_path, distance_metric)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def embed_query(self, query: str) -> List[float]:
        text = f"{self.query_prefix}{query}"
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_document(self, document: str) -> List[float]:
        text = f"{self.document_prefix}{document}"
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        texts = [f"{self.query_prefix}{q}" for q in queries]
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        texts = [f"{self.document_prefix}{d}" for d in documents]
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()


class OpenSourceModel(BaseModel):
    """
    Base class for open-source models that are not sentence-transformers.

    Subclass this and implement embed_query() and embed_document().

    Example:
        class MyModel(OpenSourceModel):
            def __init__(self):
                super().__init__("my-model")
                self.tokenizer = AutoTokenizer.from_pretrained("...")
                self.model = AutoModel.from_pretrained("...")

            def embed_query(self, query: str) -> List[float]:
                inputs = self.tokenizer(query, return_tensors="pt", truncation=True)
                outputs = self.model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

            def embed_document(self, document: str) -> List[float]:
                # Same as embed_query, or with a different prefix
                return self.embed_query(document)
    """

    def __init__(self, model_name: str, distance_metric: str = "cosine"):
        super().__init__(model_name, distance_metric)


class APIModel(BaseModel):
    """
    Base class for proprietary models accessed via API.

    Subclass this and implement embed_query() and embed_document().

    Example:
        class OpenAIModel(APIModel):
            def __init__(self):
                super().__init__("text-embedding-3-small")
                import openai
                self.client = openai.OpenAI()

            def embed_query(self, query: str) -> List[float]:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=query,
                )
                return response.data[0].embedding

            def embed_document(self, document: str) -> List[float]:
                return self.embed_query(document)
    """

    def __init__(self, model_name: str, distance_metric: str = "cosine"):
        super().__init__(model_name, distance_metric)
