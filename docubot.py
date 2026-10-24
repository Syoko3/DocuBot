"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index
- Retrieving relevant snippets (chunking)
- Retrieving semantically similar snippets with ChromaDB
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini
"""

import os
import glob
import re
import uuid

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

        # Build a retrieval index
        self.index = self.build_index(self.documents)

        # Pre-split documents into smaller, focused chunks so retrieval
        # can return a single relevant section instead of an entire file.
        self.chunks = self._build_chunks(self.documents)
        self._semantic_client = None
        self._semantic_collection = None

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
    # Index Construction
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
    # Chunking
    # -----------------------------------------------------------

    def _build_chunks(self, documents):
        """
        Split documents into heading-delimited sections, keeping each heading
        with its content. Unheaded text remains paragraph-sized. The filename
        is kept so retrieval results still point back to their source file.
        """
        chunks = []
        for filename, text in documents:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]
            section = []
            for paragraph in paragraphs:
                if not paragraph:
                    continue
                if re.match(r"^#{1,6}\s", paragraph):
                    if section:
                        chunks.append((filename, "\n\n".join(section)))
                    section = [paragraph]
                elif section:
                    section.append(paragraph)
                else:
                    chunks.append((filename, paragraph))
            if section:
                chunks.append((filename, "\n\n".join(section)))
        return chunks

    # -----------------------------------------------------------
    # Keyword Retrieval and Scoring
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Return the fraction of distinct meaningful query terms found in the text.

        The normalized keyword-overlap score is between 0.0 and 1.0. It is not
        a probability of correctness or a semantic similarity.
        """
        query_words = {w for w in _tokenize(query) if w not in STOPWORDS}
        text_tokens = set(_tokenize(text))
        if not query_words:
            return 0.0
        return len(query_words & text_tokens) / len(query_words)

    def retrieve(self, query, top_k=3):
        """
        Score every chunk, sort by relevance (descending), and return the
        top_k as (filename, chunk_text).

        If no meaningful query terms match, return an empty list so answer
        modes respond with "I do not know based on these docs."
        """
        return [
            (filename, chunk_text)
            for filename, chunk_text, _ in self.retrieve_with_scores(query, top_k)
        ]

    def retrieve_with_scores(self, query, top_k=3):
        """
        Return (filename, chunk_text, retrieval_score) tuples in ranked order.

        Scores are the fraction of distinct meaningful query terms found in
        each chunk. They range from 0.0 to 1.0 and are not probabilities of
        correctness.
        """
        if top_k <= 0:
            return []

        scored = []
        for filename, chunk_text in self.chunks:
            score = self.score_document(query, chunk_text)
            if score > 0:
                scored.append((score, filename, chunk_text))

        scored.sort(key=lambda item: item[0], reverse=True)

        if not scored:
            return []

        return [
            (filename, chunk_text, score)
            for score, filename, chunk_text in scored[:top_k]
        ]

    def rag_retrieve_with_scores(self, query, top_k=3):
        """
        Combine keyword and semantic rankings for grounded RAG context.

        Reciprocal-rank fusion combines the two rankings without treating
        their different score scales as directly comparable.
        """
        if top_k <= 0:
            return []

        candidate_count = top_k * 2
        rankings = (
            self.retrieve_with_scores(query, top_k=candidate_count),
            self.semantic_retrieve_with_scores(query, top_k=candidate_count),
        )
        fused_scores = {}
        rank_constant = 60
        for ranking in rankings:
            for rank, (filename, chunk_text, _) in enumerate(ranking, start=1):
                key = (filename, chunk_text)
                fused_scores[key] = fused_scores.get(key, 0.0) + (
                    1.0 / (rank_constant + rank)
                )

        ranked = sorted(
            (
                (score, filename, chunk_text)
                for (filename, chunk_text), score in fused_scores.items()
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            (filename, chunk_text, score)
            for score, filename, chunk_text in ranked[:top_k]
        ]

    # -----------------------------------------------------------
    # Semantic Retrieval with ChromaDB
    # -----------------------------------------------------------

    def _get_semantic_collection(self):
        """Create the in-memory Chroma collection with its default local embedder."""
        if self._semantic_collection is None and self.chunks:
            import chromadb

            self._semantic_client = chromadb.Client()
            self._semantic_collection = self._semantic_client.create_collection(
                name=f"docubot_{uuid.uuid4().hex}",
                metadata={"hnsw:space": "cosine"},
            )
            self._semantic_collection.add(
                ids=[str(index) for index in range(len(self.chunks))],
                documents=[text for _, text in self.chunks],
                metadatas=[
                    {"filename": filename} for filename, _ in self.chunks
                ],
            )
        return self._semantic_collection

    def semantic_retrieve(self, query, top_k=3):
        """Return the top semantic matches as (filename, chunk_text) tuples."""
        return [
            (filename, chunk_text)
            for filename, chunk_text, _ in self.semantic_retrieve_with_scores(
                query, top_k
            )
        ]

    def semantic_retrieve_with_scores(self, query, top_k=3):
        """
        Return semantic matches with cosine similarity scores in descending order.

        Chroma reports cosine distance (1 - cosine similarity), so convert each
        returned distance to a similarity score before exposing it.
        """
        if top_k <= 0 or not self.chunks:
            return []

        collection = self._get_semantic_collection()
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, len(self.chunks)),
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        scored = [
            (metadata["filename"], text, 1.0 - distance)
            for text, metadata, distance in zip(
                documents, metadatas, distances
            )
        ]
        return sorted(scored, key=lambda item: item[2], reverse=True)

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Retrieval only mode.
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
        RAG mode.
        Combines keyword and semantic retrieval, then asks Gemini to answer
        using only the fused snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = [
            (filename, text)
            for filename, text, _ in self.rag_retrieve_with_scores(
                query, top_k=top_k
            )
        ]

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    def answer_semantic_rag(self, query, top_k=3):
        """
        Semantic RAG Mode.
        Retrieve semantically similar snippets, then generate a grounded answer.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "Semantic RAG requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.semantic_retrieve(query, top_k=top_k)
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
