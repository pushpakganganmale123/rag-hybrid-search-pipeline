# -*- coding: utf-8 -*-
"""
RAG Hybrid Search Pipeline — Professional Streamlit Dashboard

Run:
    python scripts/ingest.py
    streamlit run streamlit_app.py
"""

import html
import re

import pandas as pd
import streamlit as st

from src.embeddings import get_embedding_provider
from src.eval import compare_chunking_strategies, run_eval
from src.generation import get_generation_provider
from src.golden_qa_loader import load_golden_qa
from src.graph import ask
from src.ingest import load_indices
from src.models import AskRequest, ChunkingStrategy
from src.reranker import get_reranker
from src.storage import list_recent_queries, save_query


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Intelligence | Hybrid Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CSS — CONSISTENT LIGHT THEME
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    color-scheme: light;
    --bg: #F5F7FB;
    --surface: #FFFFFF;
    --ink: #172033;
    --muted: #667085;
    --line: #E4E7EC;
    --primary: #5146E5;
    --primary-hover: #3930B9;
    --primary-soft: #EFEDFF;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg) !important;
    color: var(--ink) !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--ink) !important;
    font-weight: 700 !important;
}

.stApp p,
.stApp label,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p {
    color: #344054;
}

/* Sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: #FFFFFF !important;
}

[data-testid="stSidebar"] {
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: var(--ink) !important;
}

/* Hero */
.hero {
    background: linear-gradient(120deg, #191B40 0%, #38318D 55%, #6558E9 100%);
    border-radius: 18px;
    padding: 30px 34px;
    color: #FFFFFF !important;
    margin-bottom: 25px;
    overflow: hidden;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 30px;
    margin: 8px 0;
}

.hero p {
    color: #DAD9FF !important;
    font-size: 14px;
    margin-bottom: 0;
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,.13);
    border: 1px solid rgba(255,255,255,.2);
    border-radius: 30px;
    padding: 5px 12px;
    font-size: 11px;
    letter-spacing: 1px;
    color: #FFFFFF !important;
}

/* Custom metric cards */
.metric-card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 13px;
    padding: 19px;
    min-height: 116px;
    margin-bottom: 10px;
}

.metric-label {
    font-size: 12px;
    color: var(--muted) !important;
    font-weight: 600;
}

.metric-value {
    font-family: 'Manrope', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: var(--ink) !important;
    margin-top: 7px;
}

.metric-caption {
    color: var(--muted) !important;
    font-size: 11px;
    margin-top: 3px;
}

.section-heading {
    font-family: 'Manrope', sans-serif;
    font-size: 19px;
    font-weight: 800;
    color: var(--ink) !important;
    margin: 20px 0 5px;
}

.section-description {
    font-size: 12px;
    color: var(--muted) !important;
    margin-bottom: 17px;
}

.answer-box {
    background: #FFFFFF;
    color: var(--ink) !important;
    border: 1px solid var(--line);
    border-left: 4px solid var(--primary);
    padding: 22px;
    border-radius: 10px;
    line-height: 1.8;
}

/* Inputs and text areas */
.stApp input,
.stApp textarea,
.stApp [data-baseweb="input"] input,
.stApp [data-baseweb="textarea"] textarea {
    background: #FFFFFF !important;
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    border-color: #D0D5DD !important;
    caret-color: var(--primary) !important;
}

.stApp input::placeholder,
.stApp textarea::placeholder {
    color: #98A2B3 !important;
    -webkit-text-fill-color: #98A2B3 !important;
    opacity: 1 !important;
}

/* Select boxes */
.stApp [data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border-color: #D0D5DD !important;
    color: var(--ink) !important;
}

.stApp [data-baseweb="select"] input,
.stApp [data-baseweb="select"] span,
.stApp [data-baseweb="select"] div {
    color: var(--ink) !important;
}

/* Dropdown popovers */
[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="menu"] ul {
    background: #FFFFFF !important;
    color: var(--ink) !important;
}

[data-baseweb="menu"] li {
    color: var(--ink) !important;
}

[data-baseweb="menu"] li:hover {
    background: var(--primary-soft) !important;
}

/* Buttons */
.stApp .stButton > button,
.stApp .stDownloadButton > button,
.stApp [data-testid="stFormSubmitButton"] button {
    background: var(--primary) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--primary) !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    min-height: 42px;
    transition: background .2s ease, border-color .2s ease;
}

.stApp .stButton > button:hover,
.stApp .stDownloadButton > button:hover,
.stApp [data-testid="stFormSubmitButton"] button:hover {
    background: var(--primary-hover) !important;
    border-color: var(--primary-hover) !important;
    color: #FFFFFF !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid var(--line);
    background: transparent !important;
}

.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    font-size: 13px;
    font-weight: 600;
    padding: 12px 15px;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--primary) !important;
}

/* Checkbox and radio labels */
.stApp [data-testid="stCheckbox"] label,
.stApp [data-testid="stCheckbox"] p,
.stApp [data-testid="stRadio"] label,
.stApp [data-testid="stRadio"] p {
    color: var(--ink) !important;
}

/* Slider */
.stApp [data-testid="stSlider"] {
    color: var(--primary) !important;
}

/* Expander */
.stApp [data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px;
}

.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stExpander"] summary p {
    color: var(--ink) !important;
}

/* Native Streamlit metrics */
.stApp [data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px;
    padding: 14px;
}

.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricLabel"] * {
    color: var(--muted) !important;
}

.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricValue"] * {
    color: var(--ink) !important;
}

/* Dataframes and charts */
.stApp [data-testid="stDataFrame"],
.stApp [data-testid="stTable"] {
    background: #FFFFFF !important;
    border: 1px solid var(--line);
    border-radius: 10px;
}

.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p,
.stApp .stCaption {
    color: var(--muted) !important;
}

/* Alerts */
.stApp [data-testid="stAlert"] p {
    color: #344054 !important;
}

.stProgress > div > div > div {
    background: var(--primary) !important;
}

hr {
    border-color: var(--line) !important;
}

footer {
    visibility: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "last_result": None,
    "comparison_report": None,
    "hybrid_result": None,
    "dense_result": None,
    "last_question": "",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CACHED RESOURCES
# ============================================================

@st.cache_resource(show_spinner=False)
def get_index(strategy: ChunkingStrategy):
    return load_indices(strategy)


@st.cache_resource(show_spinner=False)
def get_providers():
    return (
        get_embedding_provider(),
        get_reranker(),
        get_generation_provider(),
    )


# ============================================================
# UI HELPERS
# ============================================================

def metric_card(label, value, caption=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(str(label))}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-caption">{html.escape(str(caption))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title, description=""):
    st.markdown(
        f"""
        <div class="section-heading">{html.escape(str(title))}</div>
        <div class="section-description">{html.escape(str(description))}</div>
        """,
        unsafe_allow_html=True,
    )


def render_answer_with_citations(answer_text):
    safe_text = html.escape(str(answer_text))
    return re.sub(
        r"\[(\d+)\]",
        r'<span style="color:#5146E5;font-weight:800;">[\1]</span>',
        safe_text,
    )


def render_ask_result(result, compact=False):
    st.markdown("#### Generated Answer")

    st.markdown(
        f'<div class="answer-box">{render_answer_with_citations(result.answer_text)}</div>',
        unsafe_allow_html=True,
    )

    confidence = result.confidence
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Overall Confidence", f"{confidence.composite:.0%}")
    c2.metric("Retrieval Confidence", f"{confidence.retrieval_confidence:.0%}")
    c3.metric("Citation Coverage", f"{confidence.citation_coverage:.0%}")
    c4.metric("Completeness", f"{confidence.completeness:.0%}")

    if result.citations:
        section_heading(
            "Citation Verification",
            "Evidence sources associated with the generated answer.",
        )

        citation_data = []
        for citation in result.citations:
            citation_data.append(
                {
                    "Citation": citation.marker,
                    "Source": citation.source_file,
                    "Section": citation.section_heading or "-",
                    "Supported": (
                        "Verified"
                        if citation.supported is True
                        else "Unsupported"
                        if citation.supported is False
                        else "Not checked"
                    ),
                    "Verification Note": citation.verification_note or "-",
                }
            )

        st.dataframe(
            pd.DataFrame(citation_data),
            width='stretch',
            hide_index=True,
        )

    if compact:
        return

    with st.expander(
        f"Retrieved Context | {len(result.retrieved)} chunks",
        expanded=False,
    ):
        if not result.retrieved:
            st.info("No retrieved chunks available.")

        for index, item in enumerate(result.retrieved, 1):
            dense = f"{item.dense_score:.4f}" if item.dense_score is not None else "-"
            sparse = f"{item.sparse_score:.4f}" if item.sparse_score is not None else "-"
            rerank = f"{item.rerank_score:.4f}" if item.rerank_score is not None else "-"

            st.markdown(f"**{index}. {item.chunk.source_file}**")
            if item.chunk.section_heading:
                st.caption(item.chunk.section_heading)

            a, b, c = st.columns(3)
            a.metric("Dense score", dense)
            b.metric("Sparse score", sparse)
            c.metric("Rerank score", rerank)

            st.caption(
                f"Dense rank: {item.dense_rank} | Sparse rank: {item.sparse_rank}"
            )
            chunk_text = str(item.chunk.text)
            st.text(chunk_text[:700] + ("..." if len(chunk_text) > 700 else ""))
            st.divider()

    with st.expander("Agent Execution Trace", expanded=False):
        if not result.agent_trace:
            st.info("No agent trace available.")

        for step in result.agent_trace:
            st.markdown(f"**{step.agent}**")
            st.caption(f"Execution time: {step.latency_ms:.2f} ms")
            st.write(step.summary)
            st.divider()


def get_total_chunks():
    total = 0
    for strategy in ChunkingStrategy:
        try:
            store, _ = get_index(strategy)
            total += store.count()
        except Exception:
            continue
    return total


def get_history():
    try:
        return list_recent_queries(50)
    except Exception as exc:
        st.error(f"Could not load query history: {exc}")
        return []


def get_index_checked(strategy):
    try:
        store, bm25 = get_index(strategy)
    except Exception as exc:
        st.error(f"Could not load index '{strategy.value}': {exc}")
        return None, None

    if store.count() == 0:
        st.error(
            f"No indexed documents found for '{strategy.value}'. "
            "Run `python scripts/ingest.py` first."
        )
        return None, None

    return store, bm25


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("# 🔎 RAG Intelligence")
    st.caption("Hybrid Search & Retrieval Platform")
    st.divider()

    st.markdown("### System Overview")

    try:
        sidebar_history = list_recent_queries(50)
        sidebar_chunks = get_total_chunks()
        st.metric("Indexed Chunks", sidebar_chunks)
        st.metric("Recent Queries", len(sidebar_history))
    except Exception:
        st.warning("System statistics unavailable.")

    st.divider()
    st.markdown("### Pipeline Architecture")

    pipeline_steps = [
        "Document Ingestion",
        "Text Chunking",
        "Embedding Generation",
        "Dense Retrieval",
        "BM25 Sparse Retrieval",
        "Reciprocal Rank Fusion",
        "Cross-Encoder Reranking",
        "LLM Answer Generation",
        "Citation Verification",
    ]

    for index, step in enumerate(pipeline_steps, 1):
        st.markdown(f"`{index:02}` {step}")

    st.divider()
    st.markdown("### Configuration")
    st.caption("Supported chunking strategies")

    for strategy_item in ChunkingStrategy:
        st.markdown(f"• `{strategy_item.value}`")

    st.divider()
    st.caption("RAG Hybrid Search Pipeline")
    st.caption("Streamlit Analytics Dashboard")


# ============================================================
# HERO + OVERVIEW
# ============================================================

st.markdown(
    """
    <div class="hero">
        <span class="hero-tag">AI ENGINEERING PROJECT</span>
        <h1>RAG Intelligence Dashboard</h1>
        <p>
            Explore semantic retrieval, hybrid search, document grounding,
            reranking and answer quality through one unified interface.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

history = get_history()
total_chunks = get_total_chunks()

col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card("Indexed Chunks", f"{total_chunks:,}", "Across available strategies")
with col2:
    metric_card("Recent Queries", f"{len(history):,}", "Stored query records")
with col3:
    metric_card("Chunking Strategies", len(list(ChunkingStrategy)), "Available configurations")
with col4:
    metric_card("Evaluation Dataset", "37 Cases", "Golden Q&A benchmark")


# ============================================================
# MAIN NAVIGATION
# ============================================================

tab_ask, tab_compare, tab_chunking, tab_docs, tab_history = st.tabs(
    [
        "💬 Ask",
        "⚖️ Retrieval Comparison",
        "📊 Evaluation",
        "📚 Documents",
        "🕘 History",
    ]
)


# ============================================================
# TAB 1 — ASK
# ============================================================

with tab_ask:
    section_heading(
        "Ask Your Documentation",
        "Ask natural-language questions and inspect the complete retrieval pipeline.",
    )

    example_questions = [
        "How do I declare a path parameter with a specific type?",
        "How does hybrid search combine dense and sparse retrieval?",
        "What is reciprocal rank fusion?",
        "How does a cross-encoder reranker improve retrieval?",
        "How are citations verified in the RAG pipeline?",
    ]

    selected_example = st.selectbox(
        "Choose an example question",
        ["Select a question..."] + example_questions,
        key="ask_example",
    )

    with st.form("ask_form"):
        question_default = (
            selected_example if selected_example != "Select a question..." else ""
        )

        question = st.text_area(
            "Your question",
            value=question_default,
            placeholder="Ask something about your indexed documentation...",
            height=100,
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            strategy_value = st.selectbox(
                "Chunking Strategy",
                [item.value for item in ChunkingStrategy],
                index=min(1, len(ChunkingStrategy) - 1),
            )

        with c2:
            use_hybrid = st.checkbox(
                "Enable Hybrid Retrieval",
                value=True,
                help="Combines dense vector search and sparse BM25 retrieval.",
            )

        with c3:
            top_k = st.slider("Top K Results", 1, 10, 5)

        submitted = st.form_submit_button(
            "Generate Answer →",
            type="primary",
            width='stretch',
        )

    if submitted:
        if not question.strip():
            st.warning("Enter a question before continuing.")
        else:
            selected_strategy = ChunkingStrategy(strategy_value)
            store, bm25 = get_index_checked(selected_strategy)

            if store is not None:
                try:
                    emb, reranker, gen = get_providers()

                    with st.spinner(
                        "Executing retrieval, fusion, reranking and generation..."
                    ):
                        request = AskRequest(
                            question=question.strip(),
                            chunking_strategy=selected_strategy,
                            use_hybrid=use_hybrid,
                            top_k=top_k,
                        )

                        result = ask(
                            request, store, bm25, emb, reranker, gen
                        )
                        save_query(result)

                    st.session_state.last_result = result
                    st.session_state.last_question = question.strip()
                    st.success("Answer generated successfully.")

                except Exception as exc:
                    st.error(f"Pipeline execution failed: {exc}")

    if st.session_state.last_result is not None:
        st.divider()
        st.markdown(f"**Question:** {st.session_state.last_question}")
        render_ask_result(st.session_state.last_result)


# ============================================================
# TAB 2 — HYBRID VS DENSE
# ============================================================

with tab_compare:
    section_heading(
        "Retrieval Strategy Comparison",
        "Evaluate both retrieval approaches using the same question and indexed documents.",
    )

    comparison_examples = [
        "How does hybrid retrieval improve search relevance?",
        "What is BM25 and how does it rank documents?",
        "Explain semantic search and vector embeddings.",
        "How does reciprocal rank fusion combine results?",
    ]

    compare_example = st.selectbox(
        "Choose a comparison question",
        ["Select a question..."] + comparison_examples,
        key="compare_example",
    )

    with st.form("compare_form"):
        compare_default = (
            compare_example if compare_example != "Select a question..." else ""
        )

        compare_question = st.text_area(
            "Question to compare",
            value=compare_default,
            height=85,
            placeholder="Enter a question for both retrieval systems...",
        )

        compare_strategy_value = st.selectbox(
            "Chunking Strategy",
            [item.value for item in ChunkingStrategy],
            index=min(1, len(ChunkingStrategy) - 1),
        )

        compare_submitted = st.form_submit_button(
            "Run Both Retrieval Strategies →",
            type="primary",
            width='stretch',
        )

    if compare_submitted:
        if not compare_question.strip():
            st.warning("Enter a comparison question.")
        else:
            selected_strategy = ChunkingStrategy(compare_strategy_value)
            store, bm25 = get_index_checked(selected_strategy)

            if store is not None:
                try:
                    emb, reranker, gen = get_providers()

                    with st.spinner(
                        "Running hybrid and dense-only retrieval..."
                    ):
                        hybrid_result = ask(
                            AskRequest(
                                question=compare_question.strip(),
                                chunking_strategy=selected_strategy,
                                use_hybrid=True,
                            ),
                            store, bm25, emb, reranker, gen,
                        )

                        dense_result = ask(
                            AskRequest(
                                question=compare_question.strip(),
                                chunking_strategy=selected_strategy,
                                use_hybrid=False,
                            ),
                            store, bm25, emb, reranker, gen,
                        )

                        save_query(hybrid_result)
                        save_query(dense_result)

                    st.session_state.hybrid_result = hybrid_result
                    st.session_state.dense_result = dense_result
                    st.success("Comparison completed.")

                except Exception as exc:
                    st.error(f"Comparison failed: {exc}")

    if (
        st.session_state.hybrid_result is not None
        and st.session_state.dense_result is not None
    ):
        hybrid = st.session_state.hybrid_result
        dense = st.session_state.dense_result

        st.divider()
        st.markdown("### Performance Overview")

        left, right = st.columns(2)

        with left:
            st.markdown("#### Hybrid Retrieval")
            st.metric("Composite Confidence", f"{hybrid.confidence.composite:.0%}")
            st.metric("Citation Coverage", f"{hybrid.confidence.citation_coverage:.0%}")
            st.metric("Retrieved Chunks", len(hybrid.retrieved))

        with right:
            st.markdown("#### Dense-only Retrieval")
            st.metric("Composite Confidence", f"{dense.confidence.composite:.0%}")
            st.metric("Citation Coverage", f"{dense.confidence.citation_coverage:.0%}")
            st.metric("Retrieved Chunks", len(dense.retrieved))

        st.divider()
        st.markdown("### Answer Comparison")

        left, right = st.columns(2)

        with left:
            st.markdown("#### Hybrid Answer")
            render_ask_result(hybrid, compact=True)

        with right:
            st.markdown("#### Dense-only Answer")
            render_ask_result(dense, compact=True)

        st.markdown("### Detailed Retrieval Inspection")
        left, right = st.columns(2)

        with left:
            with st.expander("Hybrid retrieved chunks"):
                for item in hybrid.retrieved:
                    st.markdown(f"**{item.chunk.source_file}**")
                    st.caption(
                        f"Dense: {item.dense_score} | "
                        f"Sparse: {item.sparse_score} | "
                        f"Rerank: {item.rerank_score}"
                    )
                    st.text(str(item.chunk.text)[:500])
                    st.divider()

        with right:
            with st.expander("Dense-only retrieved chunks"):
                for item in dense.retrieved:
                    st.markdown(f"**{item.chunk.source_file}**")
                    st.caption(
                        f"Dense: {item.dense_score} | "
                        f"Rerank: {item.rerank_score}"
                    )
                    st.text(str(item.chunk.text)[:500])
                    st.divider()


# ============================================================
# TAB 3 — CHUNKING EVALUATION
# ============================================================

with tab_chunking:
    section_heading(
        "Chunking Strategy Evaluation",
        "Benchmark all available chunking strategies against the golden Q&A dataset.",
    )

    strategies = list(ChunkingStrategy)

    a, b, c = st.columns(3)
    a.metric("Benchmark Questions", "37")
    b.metric("Strategies", len(strategies))
    c.metric("Evaluation Runs", len(strategies))

    st.info(
        "Each strategy is evaluated using the same golden Q&A dataset. "
        "Metrics include correctness, faithfulness, retrieval relevance "
        "and citation accuracy."
    )

    if st.button(
        "Run Complete Evaluation →",
        type="primary",
        width='stretch',
    ):
        try:
            dataset = load_golden_qa()
            emb, reranker, gen = get_providers()
            runs = []

            progress = st.progress(0.0)
            status = st.empty()

            for index, strategy_item in enumerate(strategies):
                status.write(
                    f"Evaluating {index + 1}/{len(strategies)}: "
                    f"{strategy_item.value}"
                )

                store, bm25 = get_index_checked(strategy_item)
                if store is None:
                    st.stop()

                evaluation_result = run_eval(
                    dataset,
                    strategy_item,
                    store,
                    bm25,
                    emb,
                    reranker,
                    gen,
                )

                runs.append(evaluation_result)
                progress.progress((index + 1) / len(strategies))

            st.session_state.comparison_report = compare_chunking_strategies(runs)
            status.success("All evaluation runs completed.")

        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

    if st.session_state.comparison_report is not None:
        report = st.session_state.comparison_report

        st.divider()
        st.markdown("### Evaluation Results")

        rows = [
            {
                "Strategy": run.chunking_strategy.value,
                "Pass Rate": run.pass_rate,
                "Correctness": run.avg_correctness,
                "Faithfulness": run.avg_faithfulness,
                "Retrieval Relevance": run.avg_retrieval_relevance,
                "Citation Accuracy": run.avg_citation_accuracy,
            }
            for run in report.runs
        ]

        results_df = pd.DataFrame(rows)
        display_df = results_df.copy()

        for column in display_df.columns[1:]:
            display_df[column] = display_df[column].map(
                lambda value: f"{value:.1%}"
            )

        st.dataframe(display_df, width='stretch', hide_index=True)

        st.markdown("### Metric Performance")
        st.bar_chart(results_df.set_index("Strategy"))

        st.markdown("### Best Performing Strategies")
        winner_items = list(report.winner_by_metric.items())
        winner_cols = st.columns(min(5, max(1, len(winner_items))))

        for index, (metric_name, winner) in enumerate(winner_items):
            with winner_cols[index % len(winner_cols)]:
                metric_card(
                    metric_name.replace("_", " ").title(),
                    winner,
                    "Best performing strategy",
                )

        if report.runs:
            best_run = max(report.runs, key=lambda item: item.pass_rate)
            st.success(
                f"Highest overall pass rate: "
                f"{best_run.chunking_strategy.value} "
                f"({best_run.pass_rate:.1%})"
            )

        st.download_button(
            "Download Evaluation Results CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="rag_evaluation_results.csv",
            mime="text/csv",
        )


# ============================================================
# TAB 4 — DOCUMENT EXPLORER
# ============================================================

with tab_docs:
    section_heading(
        "Document Explorer",
        "Explore indexed documents, chunk distribution and source content.",
    )

    selected_strategy_value = st.selectbox(
        "Select Chunking Strategy",
        [item.value for item in ChunkingStrategy],
        index=min(1, len(ChunkingStrategy) - 1),
        key="docs_strategy",
    )

    try:
        store, _ = get_index(ChunkingStrategy(selected_strategy_value))
        chunks = store.get_all_chunks()

        if not chunks:
            st.warning(
                "No documents indexed for this strategy. "
                "Run the ingestion script first."
            )
        else:
            chunks_by_file = {}
            for chunk in chunks:
                filename = chunk.source_file
                chunks_by_file[filename] = chunks_by_file.get(filename, 0) + 1

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Chunks", len(chunks))
            col2.metric("Source Documents", len(chunks_by_file))
            col3.metric("Selected Strategy", selected_strategy_value)

            st.divider()
            st.markdown("### Document Distribution")

            document_df = pd.DataFrame(
                [
                    {"Document": filename, "Chunk Count": count}
                    for filename, count in sorted(chunks_by_file.items())
                ]
            )

            st.dataframe(
                document_df,
                width='stretch',
                hide_index=True,
            )
            st.bar_chart(document_df.set_index("Document"))

            st.divider()
            st.markdown("### Browse Source Documents")

            selected_file = st.selectbox(
                "Select a document",
                sorted(chunks_by_file.keys()),
            )

            search_text = st.text_input(
                "Search within document chunks",
                placeholder="Search keywords in chunk content...",
                key="document_search",
            )

            matching_chunks = [
                chunk
                for chunk in chunks
                if chunk.source_file == selected_file
                and (
                    not search_text.strip()
                    or search_text.casefold() in str(chunk.text).casefold()
                )
            ]

            st.caption(
                f"Showing {len(matching_chunks)} matching chunks from {selected_file}"
            )

            for index, chunk in enumerate(matching_chunks, 1):
                heading = chunk.section_heading or "Unsectioned content"

                with st.expander(f"Chunk {index} | {heading}"):
                    st.markdown(f"**Source:** {chunk.source_file}")
                    st.markdown(f"**Section:** {heading}")
                    st.write(chunk.text)

    except Exception as exc:
        st.error(f"Could not load document explorer: {exc}")


# ============================================================
# TAB 5 — QUERY HISTORY
# ============================================================

with tab_history:
    section_heading(
        "Query History",
        "Review previously executed questions and their retrieval configurations.",
    )

    recent = get_history()

    if not recent:
        st.info(
            "No queries have been recorded yet. "
            "Execute a question in the Ask tab to get started."
        )
    else:
        history_rows = []

        for record in recent:
            history_rows.append(
                {
                    "ID": record.query_id,
                    "Question": record.question,
                    "Strategy": record.chunking_strategy.value,
                    "Retrieval": "Hybrid" if record.used_hybrid else "Dense-only",
                    "Confidence": record.confidence.composite,
                    "Timestamp": record.timestamp.strftime("%Y-%m-%d %H:%M"),
                }
            )

        history_df = pd.DataFrame(history_rows)

        col1, col2 = st.columns(2)
        col1.metric("Total Displayed Queries", len(history_df))
        col2.metric(
            "Average Confidence",
            f"{history_df['Confidence'].mean():.1%}",
        )

        st.divider()

        search_history = st.text_input(
            "Search query history",
            placeholder="Search questions...",
        )

        retrieval_filter = st.selectbox(
            "Retrieval Mode",
            ["All", "Hybrid", "Dense-only"],
        )

        filtered = history_df.copy()

        if search_history.strip():
            filtered = filtered[
                filtered["Question"].str.contains(
                    search_history,
                    case=False,
                    na=False,
                    regex=False,
                )
            ]

        if retrieval_filter != "All":
            filtered = filtered[filtered["Retrieval"] == retrieval_filter]

        display_history = filtered.copy()
        display_history["Confidence"] = display_history["Confidence"].map(
            lambda value: f"{value:.1%}"
        )

        st.dataframe(
            display_history,
            width='stretch',
            hide_index=True,
        )

        st.caption(
            f"Showing {len(filtered)} of {len(history_df)} recent queries."
        )

        st.download_button(
            "Export Query History CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="rag_query_history.csv",
            mime="text/csv",
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

left, right = st.columns(2)

with left:
    st.caption("RAG Intelligence Dashboard | Hybrid Search Pipeline")

with right:
    st.caption(
        "Dense Retrieval • BM25 • RRF • Reranking • Grounded Generation"
    )

