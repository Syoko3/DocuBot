"""Streamlit chat interface for DocuBot."""

from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import streamlit as st

from dataset import SAMPLE_QUERIES
from docubot import DocuBot
from llm_client import GeminiClient


load_dotenv()

# Constants for the chatbot interface
UNKNOWN_ANSWER = "I don't know based on these docs."
SUPPORTED_EXAMPLE = SAMPLE_QUERIES[2]
UNANSWERABLE_EXAMPLE = "Is there any mention of payment processing in these docs?"
API_ERROR_PREFIXES = (
    "Unable to generate an answer.",
    "API error — could not generate answer.",
)
BOT_CACHE_VERSION = 4
PACIFIC_TIME = ZoneInfo("America/Los_Angeles")
MODE_PLACEHOLDER = "Choose a mode"
CHUNKING_OPTIONS = {
    "Structure": "structure",
    "Fixed-size": "fixed_size",
    "Semantic": "semantic",
    "Recursive": "recursive",
}


@st.cache_resource
def create_bot(chunking_strategy="structure", cache_version=BOT_CACHE_VERSION):
    """Cache a DocuBot; cache_version invalidates resources after bot changes."""
    try:
        llm_client = GeminiClient()
    except RuntimeError as exc:
        return DocuBot(chunking_strategy=chunking_strategy), str(exc)
    return DocuBot(
        llm_client=llm_client,
        chunking_strategy=chunking_strategy,
    ), None


def _api_error(answer):
    """Return whether the Gemini client encoded an API failure in its result."""
    return answer.startswith(API_ERROR_PREFIXES)


def _clear_conversation():
    """Clear the chat and its input before Streamlit renders the next run."""
    st.session_state["messages"] = []
    st.session_state["question_input"] = ""
    st.session_state["mode_selection"] = MODE_PLACEHOLDER


def _answer_for_mode(bot, mode, question):
    """Run the selected mode and return its answer, sources, and error status."""
    if mode == "Naive LLM":
        answer = bot.llm_client.naive_answer_over_full_docs(
            question, bot.full_corpus_text()
        )
        return answer, [], _api_error(answer)

    semantic = mode in ("Semantic Retrieval", "Semantic RAG")
    if semantic:
        retrieve = bot.semantic_retrieve_with_scores
    elif mode == "RAG":
        retrieve = bot.rag_retrieve_with_scores
    else:
        retrieve = bot.retrieve_with_scores
    scored_sources = retrieve(question)
    sources = [(filename, text) for filename, text, _ in scored_sources]
    if mode in ("Retrieval Only", "Semantic Retrieval"):
        answer = (
            f"Retrieved {len(sources)} relevant section(s)."
            if sources
            else UNKNOWN_ANSWER
        )
        return answer, scored_sources, False

    if not sources:
        return UNKNOWN_ANSWER, scored_sources, False

    answer = bot.llm_client.answer_from_snippets(question, sources)
    return answer, scored_sources, _api_error(answer)


def _run_comparison(bot, question):
    """Compare keyword and semantic retrieval, then their RAG answers."""
    keyword_sources = bot.retrieve_with_scores(question)
    semantic_sources = bot.semantic_retrieve_with_scores(question)
    rag_sources = bot.rag_retrieve_with_scores(question)

    def generate(scored_sources):
        sources = [(filename, text) for filename, text, _ in scored_sources]
        if not sources:
            return UNKNOWN_ANSWER, False
        answer = bot.llm_client.answer_from_snippets(question, sources)
        return answer, _api_error(answer)

    rag_answer, rag_error = generate(rag_sources)
    semantic_rag_answer, semantic_rag_error = generate(semantic_sources)
    return {
        "keyword_sources": keyword_sources,
        "semantic_sources": semantic_sources,
        "rag_sources": rag_sources,
        "keyword_answer": (
            f"Retrieved {len(keyword_sources)} relevant section(s)."
            if keyword_sources else UNKNOWN_ANSWER
        ),
        "semantic_answer": (
            f"Retrieved {len(semantic_sources)} relevant section(s)."
            if semantic_sources else UNKNOWN_ANSWER
        ),
        "rag_answer": rag_answer,
        "rag_error": rag_error,
        "semantic_rag_answer": semantic_rag_answer,
        "semantic_rag_error": semantic_rag_error,
    }


def _display_sources(sources, semantic=False, hybrid=False):
    """Display retrieved text, source filename, and the retrieval score."""
    if not sources:
        st.info("No relevant sources were retrieved.")
        return

    if semantic:
        st.caption("Cosine similarity scores range from -1 to 1; higher is more similar.")
    elif hybrid:
        st.caption(
            "Reciprocal-rank fusion combines keyword and semantic rankings; "
            "these scores rank retrieved sections and are not probabilities."
        )
    else:
        st.caption(
            "Keyword scores are the fraction of distinct meaningful query "
            "terms found in a section (0.0-1.0), not probabilities of correctness."
        )
    with st.expander(f"Retrieved sources ({len(sources)})", expanded=True):
        for index, (filename, text, score) in enumerate(sources, start=1):
            score_label = (
                "Cosine similarity" if semantic
                else "Reciprocal-rank fusion" if hybrid
                else "Normalized keyword overlap"
            )
            score_value = f"{score:.4f}"
            st.markdown(
                f"**{index}. Source:** `{filename}`  \n"
                f"**{score_label}:** {score_value}"
            )
            st.write(text)


def _display_messages():
    """Render conversation history, showing evidence before generated answers."""
    for message in st.session_state["messages"]:
        with st.chat_message("user"):
            st.write(message["question"])
            if "timestamp" in message:
                st.caption(message["timestamp"])

        with st.chat_message("assistant"):
            st.caption(
                "Keyword vs Semantic Retrieval and RAG comparison"
                if message["comparison"] is not None
                else message["mode"]
            )
            if message["comparison"] is not None:
                comparison = message["comparison"]
                keyword_column, semantic_column = st.columns(2)
                with keyword_column:
                    st.markdown("**Keyword Retrieval**")
                    _display_sources(comparison["keyword_sources"])
                    st.write(comparison["keyword_answer"])
                with semantic_column:
                    st.markdown("**Semantic Retrieval**")
                    _display_sources(comparison["semantic_sources"], semantic=True)
                    st.write(comparison["semantic_answer"])

                rag_column, semantic_rag_column = st.columns(2)
                with rag_column:
                    st.markdown("**RAG (keyword + semantic retrieval)**")
                    _display_sources(comparison["rag_sources"], hybrid=True)
                    if comparison["rag_error"]:
                        st.error(comparison["rag_answer"])
                    else:
                        st.write(comparison["rag_answer"])
                with semantic_rag_column:
                    st.markdown("**Semantic RAG**")
                    _display_sources(comparison["semantic_sources"], semantic=True)
                    if comparison["semantic_rag_error"]:
                        st.error(comparison["semantic_rag_answer"])
                    else:
                        st.write(comparison["semantic_rag_answer"])
            else:
                if message["mode"] != "Naive LLM":
                    _display_sources(
                        message["sources"],
                        semantic=message["mode"] in (
                            "Semantic Retrieval",
                            "Semantic RAG",
                        ),
                        hybrid=message["mode"] == "RAG",
                    )
                if message["error"]:
                    st.error(message["answer"])
                else:
                    st.write(message["answer"])


def main():
    st.set_page_config(page_title="DocuBot", page_icon="📚")
    st.title("DocuBot")
    st.caption('''
        DocuBot is a small documentation assistant that helps answer developer questions about a codebase.
        It searches the Markdown and text files in `docs/` and can return matching excerpts or use Gemini to produce an answer.
        The user has to choose the mode and the chunking strategy, and then ask questions about the project documentation.
        The user can click also compare keyword retrieval with semantic retrieval, and compare RAG with semantic RAG, for the same question.
        The user can also click the trash can icon to clear the conversation and start over.
        The user can open the sidebar and view the loaded documents. This app requires GEMINI_API_KEY, so if the user don't have a key,
        only retrieval-only and semantic retrieval modes are available.
    ''')
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    selected_chunking = st.session_state.get(
        "chunking_strategy_selection", "Structure"
    )
    bot, llm_error = create_bot(
        CHUNKING_OPTIONS[selected_chunking],
        BOT_CACHE_VERSION,
    )

    with st.sidebar:
        st.header("Uploaded Documents")
        if bot.documents:
            documents_by_name = dict(bot.documents)
            document_options = [
                "Choose a document",
                *sorted(documents_by_name),
            ]
            selected_filename = st.selectbox(
                "Select a document to view its contents",
                document_options,
                key="document_viewer_selection",
            )
            if selected_filename == document_options[0]:
                st.caption("No documents selected.")
            else:
                st.markdown(documents_by_name[selected_filename])
        else:
            st.info("No documents were loaded.")

    if llm_error:
        st.info(
            "Gemini modes need a GEMINI_API_KEY. Retrieval-only mode is "
            "available without one; semantic retrieval is also available "
            "without one. Selecting an LLM mode will show an error."
        )

    with st.expander("Example Questions"):
        st.caption("Supported Example")
        if st.button("Use: " + SUPPORTED_EXAMPLE, key="supported_example"):
            st.session_state["question_input"] = SUPPORTED_EXAMPLE
        st.caption("Example not covered by the documents")
        if st.button("Use: " + UNANSWERABLE_EXAMPLE, key="unknown_example"):
            st.session_state["question_input"] = UNANSWERABLE_EXAMPLE

    with st.form("chat_form"):
        mode = st.selectbox(
            "Mode",
            [
                MODE_PLACEHOLDER,
                "Naive LLM",
                "Retrieval Only",
                "Semantic Retrieval",
                "RAG",
                "Semantic RAG",
            ],
            index=0,
            key="mode_selection",
            help=(
                "Semantic modes use ChromaDB's local default embedding model. "
                "The model may download on the first semantic query."
            ),
        )
        st.selectbox(
            "Chunking Strategy",
            list(CHUNKING_OPTIONS),
            key="chunking_strategy_selection",
            help=(
                "Semantic chunking uses ChromaDB's local embedding model, "
                "which may download on first use."
            ),
        )
        question = st.text_input(
            "Question",
            key="question_input",
            placeholder="Ask a documentation question...",
        )
        ask_clicked, compare_clicked, reset = st.columns([1, 3, 1])
        with ask_clicked:
            submitted = st.form_submit_button("Ask")
        with compare_clicked:
            compare = st.form_submit_button(
                "Keyword vs. Semantic Retrieval and RAG Comparison",
                disabled=bot.llm_client is None,
            )
        with reset:
            st.form_submit_button(
                "🗑️",
                help="Clear conversation",
                on_click=_clear_conversation,
            )

    if submitted or compare:
        question = question.strip()
        if not question:
            st.error("Enter a question before submitting.")
        elif not compare and mode == MODE_PLACEHOLDER:
            st.error("Choose a mode before asking a question.")
        elif compare and bot.llm_client is None:
            st.error(
                "Comparison requires Gemini. Set GEMINI_API_KEY and restart "
                "the app to enable it."
            )
        elif (
            not compare
            and mode not in (MODE_PLACEHOLDER, "Retrieval Only", "Semantic Retrieval")
            and bot.llm_client is None
        ):
            timestamp = datetime.now(PACIFIC_TIME).strftime(
                "%Y-%m-%d %I:%M:%S %p %Z"
            )
            st.session_state["messages"].append(
                {
                    "question": question,
                    "timestamp": timestamp,
                    "mode": mode,
                    "answer": (
                        "This mode requires GEMINI_API_KEY. Configure it and "
                        "restart the app to enable Gemini modes."
                    ),
                    "sources": [],
                    "error": True,
                    "comparison": None,
                }
            )
        else:
            timestamp = datetime.now(PACIFIC_TIME).strftime(
                "%Y-%m-%d %I:%M:%S %p %Z"
            )
            with st.spinner("Retrieving sources and generating an answer..."):
                if compare:
                    result = _run_comparison(bot, question)
                    message = {
                        "question": question,
                        "timestamp": timestamp,
                        "comparison": result,
                    }
                else:
                    answer, scored_sources, error = _answer_for_mode(
                        bot, mode, question
                    )
                    message = {
                        "question": question,
                        "timestamp": timestamp,
                        "mode": mode,
                        "answer": answer,
                        "sources": scored_sources,
                        "error": error,
                        "comparison": None,
                    }
            st.session_state["messages"].append(message)

    _display_messages()


if __name__ == "__main__":
    main()
