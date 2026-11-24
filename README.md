# RAG Pipeline with Hybrid Search

[![Tests](https://github.com/pushpakganganmale123/rag-hybrid-search-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/pushpakganganmale123/rag-hybrid-search-pipeline/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Streamlit-009688)

**[Live Demo](https://rag-hybrid-search-pipeline-ixdpydawwopsufco8d4vno.streamlit.app/) · [GitHub Repository](https://github.com/pushpakganganmale123/rag-hybrid-search-pipeline)**

A production-oriented Retrieval-Augmented Generation (RAG) system built over real FastAPI documentation. It combines dense vector retrieval using ChromaDB with sparse BM25 keyword search, Reciprocal Rank Fusion (RRF), reranking, grounded answer generation, citation verification, and retrieval-confidence gating.

The system is designed to do more than retrieve relevant documents: it evaluates whether retrieved evidence is sufficient to answer a question and can refuse to generate an answer when confidence is too low.

## Key Features

- **Hybrid Search:** Combines dense vector retrieval and BM25 sparse keyword retrieval.
- **Reciprocal Rank Fusion:** Merges independent retrieval results using configurable weighted RRF.
- **Multiple Chunking Strategies:** Supports fixed-size, structure-aware, and semantic chunking.
- **Reranking:** Supports lexical reranking and an LLM-as-judge implementation.
- **Grounded Generation:** Generates answers using retrieved source context with inline `[1][2]` citations.
- **Citation Verification:** Checks whether cited sources support the corresponding claims.
- **Confidence Gating:** Refuses low-confidence questions before invoking answer generation.
- **LangGraph Orchestration:** Coordinates retrieval, fusion, reranking, generation, and verification.
- **Evaluation Framework:** Uses a hand-authored golden Q&A dataset to compare retrieval and answer quality.
- **Interactive Dashboard:** Explore documents, ask questions, compare retrieval modes, and inspect evaluation results.
- **REST API:** Exposes the RAG pipeline through FastAPI endpoints.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| API | FastAPI |
| Dashboard | Streamlit |
| Vector Database | ChromaDB |
| Sparse Retrieval | BM25 (`rank_bm25`) |
| Retrieval Fusion | Reciprocal Rank Fusion |
| Orchestration | LangGraph |
| Data Validation | Pydantic |
| History Storage | SQLite |
| Testing | pytest |
| Containerization | Docker, Docker Compose |
| Embeddings | Offline hashing-based provider / OpenAI |
| Generation | Extractive mock / OpenAI / Anthropic |

## Live Demo

**[Launch the RAG Hybrid Search Dashboard](https://rag-hybrid-search-pipeline-ixdpydawwopsufco8d4vno.streamlit.app/)**

The dashboard provides:

- Ask questions against the indexed documentation.
- Inspect retrieved document chunks and citations.
- Compare hybrid search against dense-only retrieval.
- Compare fixed-size, structure-aware, and semantic chunking.
- Explore indexed documents and chunk distributions.
- Review query history and evaluation results.

## Why This Project?

Many basic RAG implementations demonstrate only document loading, embedding generation, and vector similarity search.

This project focuses on additional engineering concerns involved in building a more reliable retrieval system:

1. Dense-only retrieval can miss exact technical terms.
2. Chunking decisions should be evaluated rather than selected arbitrarily.
3. Generated citations should be checked against their source documents.
4. Retrieval confidence should influence whether generation is attempted.
5. Evaluation should include questions that the corpus cannot answer.

The goal is to make retrieval quality, answer reliability, and system behavior observable and measurable.

## The Documentation Corpus

The project uses actual FastAPI tutorial documentation rather than synthetic example text.

The `docs_corpus/` directory contains documentation fetched from the official [FastAPI repository](https://github.com/fastapi/fastapi), pinned to release tag `0.104.1`.

The fetch script:

- Retrieves the original documentation.
- Resolves `{!path/to/file.py!}` include directives.
- Inlines referenced Python code examples.
- Produces a self-contained local documentation corpus.

See `docs_corpus/MANIFEST.md` for source pages and provenance.

### Golden Evaluation Dataset

`golden_qa/qa_v1.json` contains 37 hand-authored questions grounded in the documentation corpus.

| Question category | Cases |
|---|---:|
| Straightforward lookups | 15 |
| Multi-hop questions | 8 |
| Questions with no answer in the corpus | 8 |
| Ambiguous questions | 6 |
| **Total** | **37** |

The dataset is intended to evaluate retrieval, answer correctness, faithfulness, and citation accuracy across different chunking strategies.

## Architecture

```text
                    USER QUESTION
                          |
                          v
                   LANGGRAPH PIPELINE
                          |
              +-----------+-----------+
              |                       |
              v                       v
       DENSE RETRIEVAL          SPARSE RETRIEVAL
        ChromaDB                 BM25 Search
              |                       |
              +-----------+-----------+
                          |
                          v
                RECIPROCAL RANK FUSION
                  Weighted RRF
                          |
                          v
                       RERANK
              Lexical / LLM-as-judge
                          |
                          v
                 CONFIDENCE GATE
                          |
               +----------+----------+
               |                     |
          Low confidence        Sufficient evidence
               |                     |
               v                     v
          REFUSE ANSWER       GROUNDED GENERATION
                                     |
                                     v
                              CITATION CHECK
                                     |
                                     v
                              CONFIDENCE SCORE
                                     |
                                     v
                               FINAL RESPONSE
```

### Retrieval Pipeline

**1. Dense retrieval**

ChromaDB retrieves semantically similar document chunks using vector embeddings.

**2. Sparse retrieval**

BM25 retrieves documents using keyword relevance and exact-term matching.

**3. Reciprocal Rank Fusion**

The two result sets are combined using weighted Reciprocal Rank Fusion.

The default weighting is:

- Dense retrieval: `0.7`
- Sparse retrieval: `0.3`

**4. Reranking**

Retrieved chunks are reranked using lexical overlap or an LLM-as-judge implementation.

**5. Confidence gating**

The system checks retrieval confidence before attempting generation. If the evidence does not meet the configured threshold, it returns a refusal instead of generating an unsupported answer.

**6. Grounded generation and verification**

The generation stage produces an answer with inline source references. The verification stage checks citation support and contributes to the final confidence score.

## Hybrid Retrieval: Dense vs. Sparse

Hybrid retrieval is evaluated against dense-only retrieval using the same question:

> What is an origin in the context of CORS?

The term `origin` is relatively short and generic. Dense semantic retrieval may retrieve broadly related content, while BM25 can identify exact keyword matches.

The demo script runs both retrieval approaches and displays their results side by side.

This illustrates why combining semantic similarity with lexical relevance can improve retrieval for technical documentation.

Note: ChromaDB uses approximate nearest-neighbor search, so exact ranking can vary slightly between runs.

## Chunking Strategy Comparison

The project implements three chunking strategies:

- `fixed_size`
- `structure_aware`
- `semantic`

The evaluation script runs the golden question set against each strategy and reports metrics such as pass rate, correctness, faithfulness, and citation accuracy.

### Representative Offline Evaluation

| Strategy | Pass rate | Correctness | Faithfulness | Citation accuracy |
|---|---:|---:|---:|---:|
| Fixed-size | 0.68 | 0.53 | 0.70 | 0.89 |
| Structure-aware | 0.59 | 0.47 | 0.68 | 0.84 |
| Semantic | 0.59 | 0.48 | 0.65 | 0.78 |

These are representative offline evaluation results, not universal performance guarantees.

Fixed-size chunking performed best across most metrics in this particular configuration. The offline hashing-based embedding provider is not equivalent to a production semantic embedding model, so results should be reevaluated after switching providers.

Run the comparison yourself:

```bash
python scripts/run_chunking_comparison.py
```

## Hallucination Prevention: Confidence Gating

A RAG system should not generate an answer simply because it retrieved some documents.

This project implements a retrieval-confidence gate that can stop the pipeline before generation is invoked.

The initial implementation attempted to answer every question, including questions the corpus could not answer. This resulted in unsupported answers for no-answer evaluation cases.

The confidence threshold was calibrated using the golden dataset.

The configured cutoff is `0.25` for `should_attempt_answer()`.

In the reported offline evaluation:

- No-answer accuracy improved from 0% to 75%.
- Some overlap remained between answerable and unanswerable retrieval scores.
- The threshold represents a trade-off rather than a perfect classifier.

This approach avoids unnecessary generation calls when retrieval evidence is insufficient.

See `src/confidence.py` for the implementation and calibration details.

## Switching to Real AI Providers

The project supports offline implementations and real provider-backed implementations behind common interfaces.

The offline configuration allows the project to run without API keys. Real providers can be enabled using environment variables.

Example:

```bash
export OPENAI_API_KEY=your-api-key
```

Then re-index the corpus:

```bash
python scripts/ingest.py
python scripts/run_demo.py
```

**Important:** Re-ingestion is required when switching embedding providers.

The offline embedding provider uses 256-dimensional vectors, while the configured OpenAI provider uses 1536-dimensional vectors. Existing vector collections must be rebuilt when the embedding dimension changes.

The ChromaDB collections are namespaced by chunking strategy, allowing individual strategies to be rebuilt independently.

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/pushpakganganmale123/rag-hybrid-search-pipeline.git
cd rag-hybrid-search-pipeline
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Ingest the documentation corpus

```bash
python scripts/ingest.py
```

This builds the indexes for the configured chunking strategies.

### 5. Run the demo

```bash
python scripts/run_demo.py
```

The demo covers hybrid versus dense-only retrieval, citations, a no-answer refusal, and chunking comparison.

### 6. Run tests

```bash
pytest tests/ -v
```

The test suite covers core retrieval components, chunking, embeddings, confidence scoring, citation verification, API behavior, and integration workflows.

### 7. Start the API

```bash
uvicorn src.api:app --reload
```

API documentation:

```text
http://localhost:8000/docs
```

### 8. Start the Streamlit dashboard

Open another terminal:

```bash
streamlit run streamlit_app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/v1/ask` | Ask a question |
| GET | `/v1/documents` | List indexed documents |
| POST | `/v1/ingest` | Rebuild an index |
| GET | `/v1/queries/{id}` | Retrieve a query record |
| GET | `/v1/queries` | List query history |
| GET | `/v1/eval/run` | Run evaluation |
| GET | `/v1/eval/chunking-comparison` | Compare chunking strategies |

Example request:

```json
{
  "question": "What is an origin in the context of CORS?",
  "chunking_strategy": "structure_aware",
  "use_hybrid": true
}
```

Interactive API documentation is available at `/docs` when the API is running.

## Docker Deployment

Run the application using Docker Compose:

```bash
docker compose up --build
```

The Compose configuration contains three services:

| Service | Responsibility |
|---|---|
| `init` | Ingests the documentation corpus |
| API | Runs the FastAPI service on port 8000 |
| Streamlit | Runs the dashboard on port 8501 |

The services share the initialized data volume, and the application services wait for initialization to complete.

Real provider credentials can be configured through a `.env` file.

## Important Engineering Decisions

### Why use ChromaDB instead of a mock vector database?

ChromaDB is a real local vector database that supports persistent collections and similarity search without requiring a separate hosted service.

### Why use a hashing-based offline embedding provider?

The offline provider creates deterministic bag-of-words vectors using feature hashing. It provides meaningful lexical similarity rather than returning random or constant vectors.

This makes offline retrieval and evaluation reproducible without requiring a paid embedding API.

### Why verify citations separately?

A generated citation marker does not guarantee that the cited document supports the associated claim.

The citation verification stage checks each `[N]` reference against its corresponding source content.

### Why split citation markers before checking claims?

Splitting answers on sentence punctuation can separate a citation marker from the claim it references.

The implementation instead splits on citation markers and evaluates the claim spans between them. A regression test covers this edge case.

### Why are some FastAPI endpoints synchronous?

The evaluation functions use a synchronous wrapper around the LangGraph pipeline. That wrapper invokes `asyncio.run()`.

Running it inside an already active event loop can cause a nested event-loop error. FastAPI executes synchronous route handlers in a worker thread, avoiding this conflict.

The question-answering endpoint uses the asynchronous pipeline directly.

### Why scope deduplication per source document?

Comparing every chunk against every other chunk across the entire corpus introduces unnecessary quadratic work.

Deduplication is scoped to each source file because the repeated code examples being targeted occur within individual documentation pages.

## Project Structure

```text
rag-hybrid-search-pipeline/
│
├── src/
│   ├── models.py
│   ├── document_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── bm25_index.py
│   ├── fusion.py
│   ├── reranker.py
│   ├── generation.py
│   ├── citation_verification.py
│   ├── confidence.py
│   ├── ingest.py
│   ├── graph.py
│   ├── eval.py
│   ├── storage.py
│   └── api.py
│
├── scripts/
│   ├── fetch_docs.py
│   ├── build_golden_qa.py
│   ├── ingest.py
│   ├── run_chunking_comparison.py
│   └── run_demo.py
│
├── docs_corpus/
│   └── MANIFEST.md
│
├── golden_qa/
│   └── qa_v1.json
│
├── tests/
├── streamlit_app.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Recording a Project Walkthrough

For a command-line walkthrough:

```bash
python scripts/run_demo.py
```

For a dashboard walkthrough, demonstrate:

1. Ask a technical question and inspect the generated answer and citations.
2. Open Hybrid vs. Dense-only and compare retrieval results.
3. Open Chunking Strategy Comparison and inspect the evaluation metrics.
4. Explore the indexed documentation and chunk distribution.
5. Demonstrate a question that the corpus cannot answer and show the refusal behavior.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

The documentation corpus contains content fetched from the official [FastAPI repository](https://github.com/fastapi/fastapi), which is also MIT licensed. Refer to `docs_corpus/MANIFEST.md` for source provenance and page details.
