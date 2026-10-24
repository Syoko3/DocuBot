# DocuBot

DocuBot is a small documentation assistant that helps answer developer questions about a codebase. It searches the Markdown and text files in `docs/` and can return matching excerpts or use Gemini to produce an answer. The sample documents describe API, authentication, database, and setup topics; they are static example documentation, not a live application backend.

---

## App Structure

```
DocuBot/
│
├── docs/                   # Markdown files for codebase
│   ├── API_REFERENCE.md
│   ├── AUTH.md
│   ├── DATABASE.md
│   └── SETUP.md
│
├── .env.example
├── .gitignore
├── app.py                  # Streamlit chatbot interface
├── dataset.py              # Sample Queries
├── docubot.py              # DocuBot logic
├── evaluation.py           # Evaluation of DocuBot
├── llm_client.py           # Gemini client
├── main.py                 # Command-line interface
├── README.md
└── requirements.txt
```

---

## Setup

Clone the project:

```powershell
git clone https://github.com/Syoko3/DocuBot
```

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

Copy the example file:

```powershell
cp .env.example .env
```

Then edit `.env` to include your Gemini API key (If you don't have one, create the key from the [Google AI Studio](https://aistudio.google.com/app/apikey)):

```text
GEMINI_API_KEY=your_api_key_here
```

Start the Streamlit chatbot interface:

```powershell
streamlit run app.py
```

The command-line interface is also available:

```powershell
python main.py
```

Retrieval-only and semantic retrieval modes work without an API key. Naive LLM and both RAG modes, and the retrieval-versus-RAG comparison, require `GEMINI_API_KEY`.

---

## Chat modes

| Mode | Behavior |
| --- | --- |
| **Naive LLM** | Asks Gemini a question without retrieving project documentation. Use it as an ungrounded baseline. |
| **Retrieval only** | Finds and displays matching documentation excerpts using keyword overlap. No LLM is used. |
| **Semantic retrieval** | Uses ChromaDB's default local embedding function and cosine similarity to find semantically related excerpts. No LLM is used. |
| **RAG** | Combines keyword and semantic retrieval rankings, then asks Gemini to answer using the fused excerpts. |
| **Semantic RAG** | Retrieves excerpts using ChromaDB embeddings and cosine similarity, then asks Gemini to answer using those excerpts. |

The web interface also keeps conversation history and shows each message's timestamp in Pacific Time. Users choose a mode from the **Choose a mode** option before submitting a question. The sidebar provides a **Clear conversation** action and a **Look at documents** browser for the loaded documentation. The app includes an action to compare retrieval-only and RAG results for the same question, as well as example questions including one the documents do not answer.

---

## Chunking strategies

Choose a chunking strategy in the web app's sidebar or when starting the CLI.
The default is **Structure**, which keeps Markdown headings with their content
and leaves unheaded text in paragraph-sized chunks. **Fixed-size** creates
overlapping chunks of up to 1,000 characters with a 200-character overlap.
**Semantic** groups adjacent sentences using local embedding similarity, with a
maximum chunk size of 1,000 characters. For documents with enough sentences,
it uses the lower quartile of adjacent-sentence similarities as its boundary;
very short documents use a 0.5 cosine-similarity cutoff. **Recursive** splits on
paragraphs, lines, sentence boundaries, and then spaces to keep chunks at or
below 1,000 characters where possible. Semantic chunking uses ChromaDB's
default local embedding model and may download it the first time it is selected.

All retrieval modes use the selected chunks; selecting a different strategy
rebuilds the chunk index.

---

## Retrieval scores

Keyword retrieval displays a normalized **retrieval score**: the number of distinct meaningful query terms appearing in an excerpt divided by the number of distinct meaningful query terms in the question. It ranges from `0.0` to `1.0` and is **not** a probability of correctness. Semantic retrieval displays cosine similarity, calculated as `1 - cosine distance` from an in-memory ChromaDB cosine-space collection; higher values indicate greater vector similarity, not answer correctness. ChromaDB's default local embedding model is downloaded on first semantic use.

The chunk boundaries used by both retrieval modes are controlled by the selected chunking strategy. Each mode returns up to three results. Keyword retrieval requires at least one meaningful query term to match; semantic retrieval ranks sections by embedding-vector cosine similarity. RAG combines the two rankings using reciprocal-rank fusion, while Retrieval Only remains keyword-only.

---

## Evaluation

Run the optional sample-query retrieval evaluation with:

```powershell
python evaluation.py
```

It reports whether expected source files appear in the retrieved results. The evaluation is a small demonstration, not a comprehensive quality benchmark.

---

## System notes

### Intended use and inputs

DocuBot demonstrates keyword and semantic retrieval, plus retrieval-augmented generation, on the sample project documentation. Inputs are a question, the `.md` and `.txt` files in `docs/`, and optionally a Gemini API key for LLM-powered modes.

### Retrieval design

The selected chunking strategy can preserve Markdown structure, split into overlapping fixed-size windows, group adjacent sentences by embedding similarity, or recursively split text on progressively smaller separators. Fixed-size chunks are character-based; semantic and recursive chunks are limited to 1,000 characters, and fixed-size chunks use a 200-character overlap. Semantic chunking and semantic retrieval use ChromaDB's default local embedding function. Keyword retrieval divides the count of distinct meaningful query terms found in a chunk by the total number of distinct meaningful query terms in the query. Semantic retrieval uses cosine distance, converting distance to similarity with `1 - distance`. Each retriever returns up to three chunks. RAG fuses the top keyword and semantic rankings with reciprocal-rank fusion, which does not directly compare their different score scales. Retrieval scores are not confidence or correctness probabilities.

### LLM behavior

Naive LLM mode does not use retrieved project evidence and may return plausible but unsupported answers. RAG mode fuses keyword and semantic retrieval results, then passes those excerpts to Gemini and instructs it to rely on them and to say when the evidence is insufficient. Gemini requests retry transient HTTP errors up to five attempts, with a maximum eight-second delay between attempts. If Gemini still returns a 503, its service is temporarily overloaded; try the request again shortly. RAG can still produce an incorrect answer, particularly when retrieval returns irrelevant or incomplete context.

### Limitations and responsible use

The sample corpus is small and illustrative. Paragraph splitting can separate a heading from the content it describes. Keyword matching can miss relevant text that uses different terminology, while semantic similarity can rank related but insufficient passages. Check the cited source excerpts before acting on an answer, especially for security, authentication, or configuration decisions. Prefer an explicit “I don't know” to an unsupported guess.
