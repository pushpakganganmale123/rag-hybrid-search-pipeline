
"""
Grounded answer generation with inline [1][2]-style citations.

Providers:
- MockGenerationProvider: Extractive QA without an API key.
- OpenAIGenerationProvider: LLM-based grounded answer generation.
- AnthropicGenerationProvider: LLM-based grounded answer generation.

Extractive answers copy source sentences rather than generating new claims.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import List, Tuple

from src.models import Citation, RetrievalCandidate
from src.text_utils import overlap_ratio


FALLBACK_ANSWER = (
    "I don't have enough information in the indexed "
    "documentation to answer this."
)


SYSTEM_PROMPT = """\
You are a documentation assistant. Answer the user's question using ONLY the numbered \
source passages provided below.

Rules:
- Cite the source for every factual claim using its marker, e.g. [1], [2].
- Every factual claim must have a citation.
- If the passages don't contain enough information to answer, say so explicitly.
- Do not guess or use outside knowledge.
- Do not cite a source for something it doesn't actually say.
- Do not invent technical details, values, configurations, or examples.
- Be concise and direct.
"""


class GenerationProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(
        self,
        question: str,
        candidates: List[RetrievalCandidate]
    ) -> Tuple[str, List[Citation]]:
        ...


def _build_citations(
    candidates: List[RetrievalCandidate]
) -> List[Citation]:
    """Build stable inline citations for retrieved candidates."""

    return [
        Citation(
            marker=f"[{i}]",
            chunk_id=c.chunk.id,
            source_file=c.chunk.source_file,
            section_heading=c.chunk.section_heading
        )
        for i, c in enumerate(candidates, start=1)
    ]


def _split_sentences(text: str) -> List[str]:
    """Split chunk text into clean, usable sentences."""

    if not text or not text.strip():
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+|\n+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
        and len(sentence.split()) >= 4
    ]


def _normalize_sentence(sentence: str) -> str:
    """Normalize sentences for duplicate detection."""

    return re.sub(
        r"\s+",
        " ",
        sentence.lower()
    ).strip()


class MockGenerationProvider(GenerationProvider):
    """
    Extractive question-answering provider.

    Selects the most relevant sentence from each retrieved source.
    Every answer sentence is copied from the original documentation.
    """

    name = "mock-extractive"

    MIN_RELEVANCE = 0.25
    MAX_SENTENCES = 10

    def generate(
        self,
        question: str,
        candidates: List[RetrievalCandidate]
    ) -> Tuple[str, List[Citation]]:

        if not question or not question.strip():
            return FALLBACK_ANSWER, []

        if not candidates:
            return FALLBACK_ANSWER, []

        all_citations = _build_citations(candidates)

        sentences_used = []
        used_citations = []
        seen_sentences = set()

        for citation, candidate in zip(all_citations, candidates):

            chunk_text = candidate.chunk.text

            if not chunk_text or not chunk_text.strip():
                continue

            sentences = _split_sentences(chunk_text)

            if not sentences:
                continue

            best_sentence = None
            best_score = 0.0

            for sentence in sentences:

                normalized = _normalize_sentence(sentence)

                if normalized in seen_sentences:
                    continue

                score = overlap_ratio(question, sentence)

                if score > best_score:
                    best_score = score
                    best_sentence = sentence

            if best_sentence is None:
                continue

            if best_score < self.MIN_RELEVANCE:
                continue

            sentences_used.append(
                f"{best_sentence} {citation.marker}"
            )

            used_citations.append(citation)

            seen_sentences.add(
                _normalize_sentence(best_sentence)
            )

            if len(sentences_used) >= self.MAX_SENTENCES:
                break

        if not sentences_used:
            return FALLBACK_ANSWER, []

        return " ".join(sentences_used), used_citations


def _format_context(
    candidates: List[RetrievalCandidate],
    citations: List[Citation]
) -> str:
    """Format retrieved chunks with their citation markers."""

    blocks = []

    for citation, candidate in zip(citations, candidates):

        heading = (
            f" ({candidate.chunk.section_heading})"
            if candidate.chunk.section_heading
            else ""
        )

        blocks.append(
            f"{citation.marker} Source: "
            f"{candidate.chunk.source_file}{heading}\n"
            f"{candidate.chunk.text}"
        )

    return "\n\n".join(blocks)


class OpenAIGenerationProvider(GenerationProvider):
    """Grounded answer generation using OpenAI."""

    name = "openai"

    def __init__(
        self,
        api_key: str = None,
        model: str = None
    ):
        from openai import OpenAI

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not resolved_key:
            raise ValueError(
                "OPENAI_API_KEY is required for the OpenAI provider."
            )

        self._client = OpenAI(api_key=resolved_key)

        self._model = (
            model
            or os.environ.get(
                "GENERATION_MODEL",
                "gpt-4o-mini"
            )
        )

    def generate(
        self,
        question: str,
        candidates: List[RetrievalCandidate]
    ) -> Tuple[str, List[Citation]]:

        if not candidates:
            return FALLBACK_ANSWER, []

        citations = _build_citations(candidates)

        context = _format_context(
            candidates,
            citations
        )

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": (
                        f"Sources:\n\n{context}\n\n"
                        f"Question: {question}"
                    )
                }
            ],
        )

        answer = response.choices[0].message.content

        if not answer or not answer.strip():
            return FALLBACK_ANSWER, []

        return answer.strip(), citations


class AnthropicGenerationProvider(GenerationProvider):
    """Grounded answer generation using Anthropic."""

    name = "anthropic"

    def __init__(
        self,
        api_key: str = None,
        model: str = None
    ):
        from anthropic import Anthropic

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not resolved_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required for the Anthropic provider."
            )

        self._client = Anthropic(api_key=resolved_key)

        self._model = (
            model
            or os.environ.get(
                "ANTHROPIC_MODEL",
                "claude-haiku-4-5-20251001"
            )
        )

    def generate(
        self,
        question: str,
        candidates: List[RetrievalCandidate]
    ) -> Tuple[str, List[Citation]]:

        if not candidates:
            return FALLBACK_ANSWER, []

        citations = _build_citations(candidates)

        context = _format_context(
            candidates,
            citations
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Sources:\n\n{context}\n\n"
                        f"Question: {question}"
                    )
                }
            ],
        )

        answer = "".join(
            block.text
            for block in response.content
            if hasattr(block, "text")
        )

        if not answer.strip():
            return FALLBACK_ANSWER, []

        return answer.strip(), citations


def get_generation_provider(
    name: str = None
) -> GenerationProvider:
    """
    Resolve the configured generation provider.

    Priority:
    1. Explicit provider argument.
    2. GENERATION_PROVIDER environment variable.
    3. Available OpenAI API key.
    4. Available Anthropic API key.
    5. Local extractive provider.
    """

    provider_name = (
        name or os.environ.get("GENERATION_PROVIDER", "")
    ).strip().lower()

    if provider_name == "openai":
        return OpenAIGenerationProvider()

    if provider_name == "anthropic":
        return AnthropicGenerationProvider()

    if provider_name in ("mock", "mock-extractive"):
        return MockGenerationProvider()

    if provider_name:
        raise ValueError(
            f"Unsupported GENERATION_PROVIDER: {provider_name}. "
            "Supported providers: mock, openai, anthropic."
        )

    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIGenerationProvider()

    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicGenerationProvider()

    return MockGenerationProvider()
