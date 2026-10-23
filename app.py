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
    "Unable to generate an answer. (",
    "API error — could not generate answer. (",
)
BOT_CACHE_VERSION = 2
PACIFIC_TIME = ZoneInfo("America/Los_Angeles")
MODE_PLACEHOLDER = "Choose a mode"


@st.cache_resource
def create_bot(cache_version=BOT_CACHE_VERSION):
    """Cache a DocuBot; cache_version invalidates resources after bot changes."""
    try:
        llm_client = GeminiClient()
    except RuntimeError as exc:
        return DocuBot(), str(exc)
    return DocuBot(llm_client=llm_client), None


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

    semantic = mode in ("Semantic retrieval", "Semantic RAG")
    retrieve = (
        bot.semantic_retrieve_with_scores if semantic
        else bot.retrieve_with_scores
    )
    scored_sources = retrieve(question)
    sources = [(filename, text) for filename, text, _ in scored_sources]
    if mode in ("Retrieval only", "Semantic retrieval"):
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
    """Run retrieval-only and RAG using the same question and sources."""
    scored_sources = bot.retrieve_with_scores(question)
    sources = [(filename, text) for filename, text, _ in scored_sources]
    if not scored_sources:
        return {
            "sources": [],
            "retrieval_answer": UNKNOWN_ANSWER,
            "rag_answer": UNKNOWN_ANSWER,
            "rag_error": False,
        }

    rag_answer = bot.llm_client.answer_from_snippets(question, sources)
    return {
        "sources": scored_sources,
        "retrieval_answer": f"Retrieved {len(sources)} relevant section(s).",
        "rag_answer": rag_answer,
        "rag_error": _api_error(rag_answer),
    }


def _display_sources(sources, semantic=False):
    """Display retrieved text, source filename, and the retrieval score."""
    if not sources:
        st.info("No relevant sources were retrieved.")
        return

    if semantic:
        st.caption("Cosine similarity scores range from -1 to 1; higher is more similar.")
    else:
        st.caption(
            "Keyword scores are the fraction of distinct meaningful query "
            "terms found in a section (0.0-1.0), not probabilities of correctness."
        )
    with st.expander(f"Retrieved sources ({len(sources)})", expanded=True):
        for index, (filename, text, score) in enumerate(sources, start=1):
            score_label = (
                "Cosine similarity" if semantic
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
                "Retrieval-only vs RAG comparison"
                if message["comparison"] is not None
                else message["mode"]
            )
            if message["comparison"] is not None:
                comparison = message["comparison"]
                _display_sources(comparison["sources"])
                retrieval_column, rag_column = st.columns(2)
                with retrieval_column:
                    st.markdown("**Retrieval only**")
                    st.write(comparison["retrieval_answer"])
                with rag_column:
                    st.markdown("**RAG**")
                    if comparison["rag_error"]:
                        st.error(comparison["rag_answer"])
                    else:
                        st.write(comparison["rag_answer"])
            else:
                if message["mode"] != "Naive LLM":
                    _display_sources(
                        message["sources"],
                        semantic=message["mode"] in (
                            "Semantic retrieval",
                            "Semantic RAG",
                        ),
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
        Ask questions about the project documentation. You can click the "Clear Conversation" button to clear the conversation.
        You can also click the "Compare retrieval-only vs RAG" button to see how the two modes differ.
        Open the sidebar to view the loaded documents. If you don't have a GEMINI_API_KEY, only retrieval-only and semantic retrieval modes are available.
    ''')
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    bot, llm_error = create_bot(BOT_CACHE_VERSION)
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
                with st.expander(f"{selected_filename}", expanded=True):
                    st.markdown(documents_by_name[selected_filename])
        else:
            st.info("No documents were loaded.")

    if llm_error:
        st.info(
            "Gemini modes need a GEMINI_API_KEY. Retrieval-only mode is "
            "available without one; semantic retrieval is also available "
            "without one. Selecting an LLM mode will show an error."
        )

    with st.expander("Example questions"):
        st.caption("Supported example")
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
                "Retrieval only",
                "Semantic retrieval",
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
        question = st.text_input(
            "Question",
            key="question_input",
            placeholder="Ask a documentation question...",
        )
        ask_clicked, compare_clicked, reset = st.columns(3)
        with ask_clicked:
            submitted = st.form_submit_button("Ask")
        with compare_clicked:
            compare = st.form_submit_button(
                "Compare retrieval-only vs RAG",
                disabled=bot.llm_client is None,
            )
        with reset:
            st.form_submit_button(
                "Clear conversation",
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
            and mode not in (MODE_PLACEHOLDER, "Retrieval only", "Semantic retrieval")
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
