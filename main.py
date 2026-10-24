"""
CLI runner for the DocuBot.

Supports three modes:
1. Naive LLM generation over all docs
2. Retrieval only
3. Semantic retrieval only (ChromaDB embeddings)
4. RAG: hybrid keyword and semantic retrieval plus LLM generation
5. Semantic RAG: semantic retrieval plus LLM generation
You can run this script with `python main.py` or `python -m docubot.main`.
"""

from dotenv import load_dotenv
load_dotenv()

from docubot import CHUNKING_STRATEGIES, DocuBot
from llm_client import GeminiClient
from dataset import SAMPLE_QUERIES


def try_create_llm_client():
    """
    Tries to create a GeminiClient.
    Returns (llm_client, has_llm: bool).
    """
    try:
        client = GeminiClient()
        return client, True
    except RuntimeError as exc:
        print("Warning: LLM features are disabled.")
        print(f"Reason: {exc}")
        print("You can still run retrieval only mode.\n")
        return None, False


def choose_mode(has_llm):
    """
    Asks the user which mode to run.
    Returns "1" through "5", or "q".
    """
    print("Choose a mode:")
    if has_llm:
        print("  1) Naive LLM (No Retrieval)")
    else:
        print("  1) Naive LLM (Unavailable, no GEMINI_API_KEY)")
    print("  2) Retrieval Only (No LLM)")
    print("  3) Semantic Retrieval Only (ChromaDB Embeddings)")
    if has_llm:
        print("  4) RAG (Retrieval + LLM, Keyword + Semantic Hybrid)")
        print("  5) Semantic RAG (ChromaDB Retrieval + LLM)")
    else:
        print("  4) RAG (Unavailable, No GEMINI_API_KEY)")
        print("  5) Semantic RAG (Unavailable, No GEMINI_API_KEY)")
    print("  0) Quit")

    choice = input("Enter choice: ").strip().lower()
    return choice


def choose_chunking_strategy():
    """Ask which strategy DocuBot should use to build its retrieval chunks."""
    print("\nChoose a chunking strategy:")
    for index, strategy in enumerate(CHUNKING_STRATEGIES, start=1):
        print(f"  {index}) {strategy.replace('_', ' ').title()}")
    choice = input("Enter choice (default 1): ").strip()
    if not choice:
        return CHUNKING_STRATEGIES[0]
    if choice.isdigit() and 1 <= int(choice) <= len(CHUNKING_STRATEGIES):
        return CHUNKING_STRATEGIES[int(choice) - 1]
    print("Unknown choice; using structure chunking.\n")
    return CHUNKING_STRATEGIES[0]


def get_query_or_use_samples():
    """
    Ask the user if they want to run all sample queries or a single custom query.

    Returns:
        queries: list of strings
        label: short description of the source of queries
    """
    print("\nPress Enter to run built in sample queries.")
    custom = input("Or type a single custom query: ").strip()

    if custom:
        return [custom], "custom query"
    else:
        return SAMPLE_QUERIES, "sample queries"


def run_naive_llm_mode(bot, has_llm):
    """Mode 1: Naive LLM generation over the full docs corpus."""
    if not has_llm or bot.llm_client is None:
        print("\nNaive LLM mode is not available (no GEMINI_API_KEY).\n")
        return

    queries, label = get_query_or_use_samples()
    print(f"\nRunning naive LLM mode on {label}...\n")

    all_text = bot.full_corpus_text()

    for query in queries:
        print("=" * 60)
        print(f"Question: {query}\n")
        answer = bot.llm_client.naive_answer_over_full_docs(query, all_text)
        print("Answer:")
        print(answer)
        print()


def run_retrieval_only_mode(bot):
    """Mode 2: Retrieval only answers. No LLM involved."""
    queries, label = get_query_or_use_samples()
    print(f"\nRunning retrieval only mode on {label}...\n")

    for query in queries:
        print("=" * 60)
        print(f"Question: {query}\n")
        answer = bot.answer_retrieval_only(query)
        print("Retrieved snippets:")
        print(answer)
        print()


def run_semantic_retrieval_mode(bot):
    """Mode 3: Retrieve relevant snippets using local embedding vectors."""
    queries, label = get_query_or_use_samples()
    print(f"\nRunning semantic retrieval mode on {label}...\n")

    for query in queries:
        print("=" * 60)
        print(f"Question: {query}\n")
        for filename, text, score in bot.semantic_retrieve_with_scores(query):
            print(f"[{filename}] cosine similarity: {score:.4f}")
            print(text)
            print()


def run_rag_mode(bot, has_llm):
    """Mode 4: Hybrid retrieval plus Gemini answer generation."""
    if not has_llm or bot.llm_client is None:
        print("\nRAG mode is not available (no GEMINI_API_KEY).\n")
        return

    queries, label = get_query_or_use_samples()
    print(f"\nRunning RAG mode on {label}...\n")

    for query in queries:
        print("=" * 60)
        print(f"Question: {query}\n")
        answer = bot.answer_rag(query)
        print("Answer:")
        print(answer)
        print()

def run_semantic_rag_mode(bot, has_llm):
    """Mode 5: Semantic retrieval followed by Gemini answer generation."""
    if not has_llm or bot.llm_client is None:
        print("\nSemantic RAG is not available (no GEMINI_API_KEY).\n")
        return

    queries, label = get_query_or_use_samples()
    print(f"\nRunning semantic RAG mode on {label}...\n")

    for query in queries:
        print("=" * 60)
        print(f"Question: {query}\n")
        print("Answer:")
        print(bot.answer_semantic_rag(query))
        print()


def main():
    print("Welcome to DocuBot command line interface.")
    print("==========================================\n")

    llm_client, has_llm = try_create_llm_client()

    while True:
        choice = choose_mode(has_llm)

        if choice == "0":
            print("\nGoodbye.")
            break
        elif choice in ("1", "2", "3", "4", "5"):
            chunking_strategy = choose_chunking_strategy()
            bot = DocuBot(
                llm_client=llm_client,
                chunking_strategy=chunking_strategy,
            )
            if choice == "1":
                run_naive_llm_mode(bot, has_llm)
            elif choice == "2":
                run_retrieval_only_mode(bot)
            elif choice == "3":
                run_semantic_retrieval_mode(bot)
            elif choice == "4":
                run_rag_mode(bot, has_llm)
            elif choice == "5":
                run_semantic_rag_mode(bot, has_llm)
        else:
            print("\nUnknown choice. Please pick 1, 2, 3, 4, 5, or 0.\n")


if __name__ == "__main__":
    main()
