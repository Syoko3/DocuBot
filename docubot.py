"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1 + Phase 3 chunking)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)

This is the reference *solution*: the three stubbed methods (`build_index`,
`score_document`, `retrieve`) are implemented, and the Phase 3 refinements are
included — retrieval works on paragraph-sized chunks (not whole files) and a
guardrail refuses to answer when nothing relevant is found.
"""

import os
import glob
import re

# Very common words that carry little retrieval signal. Dropping them from the
# query keeps a question like "Where is the auth token generated?" focused on
# the words that matter ("auth", "token", "generated").
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on",
    "for", "and", "or", "do", "does", "did", "how", "what", "which", "where",
    "when", "who", "why", "i", "you", "it", "this", "that", "these", "those",
    "there", "any", "mention", "me", "my", "can", "with", "from", "into", "as",
    # Words about the corpus itself, not its content — they add noise because
    # "docs"/"documentation" literally appear throughout the documentation.
    "docs", "doc", "documentation",
}


def _tokenize(text):
    """Lowercase and split into alphanumeric word tokens (punctuation removed)."""
    return re.findall(r"[a-z0-9_]+", text.lower())


class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

        # Phase 3: pre-split documents into smaller, focused chunks so retrieval
        # can return a single relevant section instead of an entire file.
        self.chunks = self._build_chunks(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        Build a tiny inverted index mapping lowercase words to the documents
        they appear in, e.g. {"token": ["AUTH.md", "API_REFERENCE.md"]}.
        """
        index = {}
        for filename, text in documents:
            for token in set(_tokenize(text)):
                index.setdefault(token, [])
                if filename not in index[token]:
                    index[token].append(filename)
        return index

    # -----------------------------------------------------------
    # Chunking (Phase 3)
    # -----------------------------------------------------------

    def _build_chunks(self, documents):
        """
        Split each document into paragraph-sized chunks (separated by blank
        lines). Returns a list of (filename, chunk_text). The filename is kept
        so retrieval results still point back to their source file.
        """
        chunks = []
        for filename, text in documents:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]
            for paragraph in paragraphs:
                if paragraph:
                    chunks.append((filename, paragraph))
        return chunks

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Return a relevance score for how well the text matches the query.

        Counts how many of the query's meaningful words (stopwords removed)
        appear as whole tokens in the text. More overlap -> higher score.
        Using whole-token matching (rather than substring) keeps the scoring
        precise, so a doc that merely mentions "token" a lot doesn't drown out
        the doc that actually answers a token question.
        """
        query_words = [w for w in _tokenize(query) if w not in STOPWORDS]
        text_tokens = set(_tokenize(text))
        return sum(1 for word in query_words if word in text_tokens)

    # Minimum score the best chunk must reach for DocuBot to answer at all.
    # With corpus-meta words ("docs") filtered out of the query, an off-topic
    # question like "payment processing in these docs" matches nothing and is
    # refused; any genuine keyword hit clears this bar.
    MIN_CONFIDENT_SCORE = 1

    def retrieve(self, query, top_k=3):
        """
        Score every chunk, sort by relevance (descending), and return the
        top_k as (filename, chunk_text).

        Guardrail (Phase 3): if the best chunk scores below
        MIN_CONFIDENT_SCORE (i.e. 0 or 1 query words matched), return an empty
        list so the answer modes respond with "I do not know based on these
        docs." rather than serving a weak, off-topic match.
        """
        scored = []
        for filename, chunk_text in self.chunks:
            score = self.score_document(query, chunk_text)
            if score > 0:
                scored.append((score, filename, chunk_text))

        scored.sort(key=lambda item: item[0], reverse=True)

        if not scored or scored[0][0] < self.MIN_CONFIDENT_SCORE:
            return []

        return [(filename, chunk_text) for _, filename, chunk_text in scored[:top_k]]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
