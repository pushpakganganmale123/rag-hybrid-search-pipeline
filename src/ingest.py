
import math
from typing import List, Optional

from src.bm25_index import BM25Index
from src.chunking import chunk_document
from src.document_loader import load_corpus
from src.embeddings import EmbeddingProvider
from src.models import ChunkingStrategy, DocumentChunk
from src.vector_store import VectorStore


DUPLICATE_SIMILARITY_THRESHOLD = 0.95


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two embedding vectors."""

    if len(a) != len(b):
        raise ValueError("Embedding dimensions do not match")

    dot = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def deduplicate_within_file(
    chunks: List[DocumentChunk],
    embeddings: List[List[float]],
    strategy: Optional[ChunkingStrategy] = None,
):
    """
    Remove near-duplicate chunks within each source file.

    Semantic chunks are preserved without deduplication because
    high embedding similarity does not necessarily mean identical
    semantic content.
    """

    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count mismatch")

    if strategy == ChunkingStrategy.SEMANTIC:
        return chunks, embeddings, 0

    kept_chunks = []
    kept_embeddings = []
    skipped = 0

    by_file = {}

    for chunk, emb in zip(chunks, embeddings):
        seen = by_file.setdefault(chunk.source_file, [])

        duplicate = any(
            cosine_similarity(emb, other)
            >= DUPLICATE_SIMILARITY_THRESHOLD
            for other in seen
        )

        if duplicate:
            skipped += 1
            continue

        seen.append(emb)

        kept_chunks.append(chunk)
        kept_embeddings.append(emb)

    return kept_chunks, kept_embeddings, skipped


def load_indices(strategy: ChunkingStrategy):
    """
    Load an existing persisted Chroma collection
    and rebuild its in-memory BM25 index.
    """

    vector_store = VectorStore(strategy)

    bm25_index = BM25Index()

    bm25_index.build(vector_store.get_all_chunks())

    return vector_store, bm25_index


def load_or_ingest_indices(
    strategy: ChunkingStrategy,
    embedding_provider: Optional[EmbeddingProvider] = None,
):
    """
    Load an existing index.

    If the selected collection is empty, ingest the corpus.
    Existing populated collections are never rebuilt here.
    """

    vector_store, bm25_index = load_indices(strategy)

    if vector_store.count() == 0:

        if embedding_provider is None:
            from src.embeddings import get_embedding_provider

            embedding_provider = get_embedding_provider()

        ingest_corpus(
            strategy,
            embedding_provider,
            vector_store,
            bm25_index,
        )

    return vector_store, bm25_index


def ingest_corpus(
    strategy: ChunkingStrategy,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    bm25_index: BM25Index,
) -> dict:
    """
    Ingest documents for one selected chunking strategy.

    Only the selected strategy's ChromaDB collection
    is cleared and rebuilt.
    """

    documents = load_corpus()

    all_chunks = []

    # Generate chunks for the selected strategy.
    for doc in documents:

        kwargs = {}

        if strategy == ChunkingStrategy.SEMANTIC:
            kwargs["embedding_provider"] = embedding_provider

        chunks = chunk_document(
            doc,
            strategy,
            **kwargs,
        )

        all_chunks.extend(chunks)

    # Generate embeddings.
    embeddings = embedding_provider.embed(
        [chunk.text for chunk in all_chunks]
    )

    if len(embeddings) != len(all_chunks):
        raise ValueError(
            "Embedding provider returned an unexpected number of embeddings"
        )

    # Deduplicate according to the selected strategy.
    kept_chunks, kept_embeddings, skipped = deduplicate_within_file(
        all_chunks,
        embeddings,
        strategy,
    )

    # Do not clear a collection if there is no data.
    if not kept_chunks:
        raise ValueError(
            "No chunks available. Existing collection was not modified."
        )

    # Clear ONLY the selected strategy collection.
    vector_store.clear()

    # Store the new chunks and embeddings.
    vector_store.add(
        kept_chunks,
        kept_embeddings,
    )

    # Rebuild the corresponding BM25 index.
    bm25_index.build(kept_chunks)

    return {
        "strategy": strategy.value,
        "documents": len(documents),
        "chunks_before_dedup": len(all_chunks),
        "chunks_indexed": len(kept_chunks),
        "duplicates_skipped": skipped,
        "collection_count": vector_store.count(),
    }
