
"""Evaluation framework: runs the golden Q&A set through the real pipeline
and scores four dimensions:
- Answer correctness
- Faithfulness
- Retrieval relevance
- Citation accuracy

The chunking strategy comparison runs the same evaluation set against
each strategy's indices and reports the winner for every metric.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

from src.bm25_index import BM25Index
from src.citation_verification import claims_by_marker
from src.embeddings import EmbeddingProvider
from src.generation import GenerationProvider
from src.graph import ask
from src.models import (
    AskRequest,
    ChunkingComparisonReport,
    ChunkingStrategy,
    Difficulty,
    EvalCaseResult,
    EvalRunSummary,
    GoldenDataset,
)
from src.reranker import Reranker
from src.text_utils import overlap_ratio
from src.vector_store import VectorStore


REFUSAL_PHRASE = "don't have enough information"
CORRECTNESS_PASS_THRESHOLD = 0.35

EVALUATION_METRICS = [
    "pass_rate",
    "avg_correctness",
    "avg_faithfulness",
    "avg_retrieval_relevance",
    "avg_citation_accuracy",
]


def _score_correctness(
    qa_difficulty: Difficulty,
    generated: str,
    expected: str,
) -> float:
    """Score generated answer similarity against the expected answer."""

    if qa_difficulty == Difficulty.NO_ANSWER:
        return (
            1.0
            if REFUSAL_PHRASE in generated.lower()
            else 0.0
        )

    return min(overlap_ratio(expected, generated) * 1.3, 1.0)


def _score_retrieval_relevance(
    qa_source_files: List[str],
    retrieved_source_files: List[str],
) -> float:
    """Measure how many expected source files were retrieved."""

    if not qa_source_files:
        return 1.0

    hits = sum(
        1
        for source_file in qa_source_files
        if source_file in retrieved_source_files
    )

    return hits / len(qa_source_files)


def _score_faithfulness(
    answer_text: str,
    num_citations_used: int,
) -> float:
    """Measure the fraction of claim spans with citations.

    This measures citation coverage, not citation correctness.
    Citation accuracy is evaluated separately.
    """

    claims = claims_by_marker(answer_text)

    non_empty_claims = [
        claim
        for claim in claims.values()
        if claim
    ]

    total_spans = len(non_empty_claims)

    if total_spans == 0:
        return 0.0

    return min(num_citations_used / total_spans, 1.0)


def run_eval(
    dataset: GoldenDataset,
    chunking_strategy: ChunkingStrategy,
    vector_store: VectorStore,
    bm25_index: BM25Index,
    embedding_provider: EmbeddingProvider,
    reranker: Reranker,
    generation_provider: GenerationProvider,
) -> EvalRunSummary:
    """Run all golden Q&A cases through the actual RAG pipeline."""

    results: List[EvalCaseResult] = []

    for qa in dataset.cases:
        request = AskRequest(
            question=qa.question,
            chunking_strategy=chunking_strategy,
        )

        result = ask(
            request,
            vector_store,
            bm25_index,
            embedding_provider,
            reranker,
            generation_provider,
        )

        retrieved_files = [
            chunk.chunk.source_file
            for chunk in result.retrieved
        ]

        used_citations = [
            citation
            for citation in result.citations
            if citation.supported is not None
        ]

        correctness = _score_correctness(
            qa.difficulty,
            result.answer_text,
            qa.expected_answer,
        )

        faithfulness = _score_faithfulness(
            result.answer_text,
            len(used_citations),
        )

        retrieval_relevance = _score_retrieval_relevance(
            qa.source_files,
            retrieved_files,
        )

        if used_citations:
            citation_accuracy = (
                sum(
                    1
                    for citation in used_citations
                    if citation.supported
                )
                / len(used_citations)
            )
        else:
            citation_accuracy = (
                1.0
                if qa.difficulty == Difficulty.NO_ANSWER
                else 0.0
            )

        passed = (
            correctness >= CORRECTNESS_PASS_THRESHOLD
            and (
                not used_citations
                or citation_accuracy >= 0.5
            )
        )

        results.append(
            EvalCaseResult(
                qa_id=qa.id,
                difficulty=qa.difficulty,
                question=qa.question,
                generated_answer=result.answer_text,
                expected_answer=qa.expected_answer,
                correctness_score=round(correctness, 3),
                faithfulness_score=round(faithfulness, 3),
                retrieval_relevance_score=round(
                    retrieval_relevance,
                    3,
                ),
                citation_accuracy_score=round(
                    citation_accuracy,
                    3,
                ),
                passed=passed,
            )
        )

    total = len(results)

    per_difficulty: dict = {}

    for difficulty in Difficulty:
        cases = [
            result
            for result in results
            if result.difficulty == difficulty
        ]

        if cases:
            passed_count = sum(
                1
                for case in cases
                if case.passed
            )

            per_difficulty[difficulty.value] = (
                passed_count / len(cases)
            )

    return EvalRunSummary(
        run_id=f"eval-{uuid.uuid4().hex[:10]}",
        chunking_strategy=chunking_strategy,
        timestamp=datetime.now(timezone.utc),
        dataset_version=dataset.dataset_version,
        results=results,
        pass_rate=(
            sum(1 for result in results if result.passed) / total
            if total
            else 0.0
        ),
        avg_correctness=(
            sum(result.correctness_score for result in results) / total
            if total
            else 0.0
        ),
        avg_faithfulness=(
            sum(result.faithfulness_score for result in results) / total
            if total
            else 0.0
        ),
        avg_retrieval_relevance=(
            sum(
                result.retrieval_relevance_score
                for result in results
            )
            / total
            if total
            else 0.0
        ),
        avg_citation_accuracy=(
            sum(
                result.citation_accuracy_score
                for result in results
            )
            / total
            if total
            else 0.0
        ),
        per_difficulty_pass_rate=per_difficulty,
    )


def compare_chunking_strategies(
    runs: List[EvalRunSummary],
) -> ChunkingComparisonReport:
    """Compare evaluation runs and identify the winner per metric."""

    if not runs:
        raise ValueError(
            "Cannot compare chunking strategies: "
            "no evaluation runs were provided."
        )

    winner_by_metric = {}

    for metric in EVALUATION_METRICS:
        best = max(
            runs,
            key=lambda run: getattr(run, metric),
        )

        winner_by_metric[metric] = best.chunking_strategy.value

    return ChunkingComparisonReport(
        runs=runs,
        winner_by_metric=winner_by_metric,
    )
