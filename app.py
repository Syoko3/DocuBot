"""Streamlit chat interface for DocuBot."""

from dotenv import load_dotenv
import streamlit as st

from dataset import SAMPLE_QUERIES
from docubot import DocuBot
from llm_client import GeminiClient


load_dotenv()

UNKNOWN_ANSWER = "I don't know based on these docs."
SUPPORTED_EXAMPLE = SAMPLE_QUERIES[2]
UNANSWERABLE_EXAMPLE = "Is there any mention of payment processing in these docs?"
API_ERROR_PREFIXES = (
    "Unable to generate an answer. (",
    "API error — could not generate answer. (",
)


@st.cache_resource
def create_bot():
    """Create a shared DocuBot, enabling LLM modes when an API key is set."""
    try:
        llm_client = GeminiClient()
    except RuntimeError as exc:
        return DocuBot(), str(exc)
    return DocuBot(llm_client=llm_client), None


def _api_error(answer):
    """Return whether the Gemini client encoded an API failure in its result."""
    return answer.startswith(API_ERROR_PREFIXES)


def _answer_for_mode(bot, mode, question):
    """Run the selected mode and return its answer, sources, and error status."""
    if mode == "Naive LLM":
        answer = bot.llm_client.naive_answer_over_full_docs(
            question, bot.full_corpus_text()
        )
        return answer, [], _api_error(answer)

    sources = bot.retrieve(question)
    if mode == "Retrieval only":
        answer = (
            f"Retrieved {len(sources)} relevant section(s)."
            if sources
            else UNKNOWN_ANSWER
        )
        return answer, sources, False

    if not sources:
        return UNKNOWN_ANSWER, sources, False

    answer = bot.answer_rag(question)
    return answer, sources, _api_error(answer)


def _run_comparison(bot, question):
    """Run retrieval-only and RAG using the same question and sources."""
    sources = bot.retrieve(question)
    if not sources:
        return {
            "sources": [],
            "retrieval_answer": UNKNOWN_ANSWER,
            "rag_answer": UNKNOWN_ANSWER,
            "rag_error": False,
        }

    rag_answer = bot.answer_rag(question)
    return {
        "sources": sources,
        "retrieval_answer": f"Retrieved {len(sources)} relevant section(s).",
        "rag_answer": rag_answer,
        "rag_error": _api_error(rag_answer),
    }


def _display_sources(sources):
    """Display retrieved section text with its filename."""
    if not sources:
        st.info("No relevant sources were retrieved.")
        return

    with st.expander(f"Retrieved sources ({len(sources)})", expanded=True):
        for index, (filename, text) in enumerate(sources, start=1):
            st.markdown(f"**{index}. Source:** `{filename}`")
            st.write(text)


def _display_messages():
    """Render conversation history, showing evidence before generated answers."""
    for message in st.session_state["messages"]:
        with st.chat_message("user"):
            st.write(message["question"])

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
                    _display_sources(message["sources"])
                if message["error"]:
                    st.error(message["answer"])
                else:
                    st.write(message["answer"])


def main():
    st.set_page_config(page_title="DocuBot", page_icon="📚")
    st.title("DocuBot")
    st.caption("Ask questions about the project documentation.")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    bot, llm_error = create_bot()
    if llm_error:
        st.info(
            "Gemini modes need a GEMINI_API_KEY. Retrieval-only mode is "
            "available without one; selecting an LLM mode will show an error."
        )

    if st.button("Clear conversation"):
        st.session_state["messages"] = []
        st.session_state["question_input"] = ""
        st.rerun()

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
            ["Naive LLM", "Retrieval only", "RAG"],
            index=1,
        )
        question = st.text_input(
            "Question",
            key="question_input",
            placeholder="Ask a documentation question...",
        )
        ask_clicked, compare_clicked = st.columns(2)
        with ask_clicked:
            submitted = st.form_submit_button("Ask")
        with compare_clicked:
            compare = st.form_submit_button(
                "Compare retrieval-only vs RAG",
                disabled=bot.llm_client is None,
            )

    if submitted or compare:
        question = question.strip()
        if not question:
            st.error("Enter a question before submitting.")
        elif compare and bot.llm_client is None:
            st.error(
                "Comparison requires Gemini. Set GEMINI_API_KEY and restart "
                "the app to enable it."
            )
        elif not compare and mode != "Retrieval only" and bot.llm_client is None:
            st.session_state["messages"].append(
                {
                    "question": question,
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
            with st.spinner("Retrieving sources and generating an answer..."):
                if compare:
                    result = _run_comparison(bot, question)
                    message = {
                        "question": question,
                        "comparison": result,
                    }
                else:
                    answer, sources, error = _answer_for_mode(
                        bot, mode, question
                    )
                    message = {
                        "question": question,
                        "mode": mode,
                        "answer": answer,
                        "sources": sources,
                        "error": error,
                        "comparison": None,
                    }
            st.session_state["messages"].append(message)

    _display_messages()


if __name__ == "__main__":
    main()
