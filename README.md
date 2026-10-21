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
├── .env.example            # Make a copy of this
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

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

Copy the example file:

```powershell
cp .env.example .env
```

Then edit `.env` to include your Gemini API key:

```text
GEMINI_API_KEY=your_api_key_here
```

Start the web chat:

```powershell
streamlit run app.py
```

The command-line interface is also available:

```powershell
python main.py
```

Retrieval-only mode works without an API key. Naive LLM and RAG modes, and the retrieval-versus-RAG comparison, require `GEMINI_API_KEY`.

---

## Chat modes

| Mode | Behavior |
| --- | --- |
| **Naive LLM** | Asks Gemini a question without retrieving project documentation. Use it as an ungrounded baseline. |
| **Retrieval only** | Finds and displays matching documentation excerpts and their source filenames. No LLM is used. |
| **RAG** | Retrieves documentation excerpts, then asks Gemini to answer using those excerpts. |

The web interface also keeps conversation history, can clear it, and includes an action to compare retrieval-only and RAG results for the same question. Example questions are provided in the app, including one the documents do not answer.

---

## Retrieval scores

Each retrieved excerpt displays a **retrieval score**: the number of distinct meaningful query terms that appear in that excerpt. This raw keyword-overlap count ranks excerpts; it is **not** a probability that an excerpt or answer is correct. Stopwords are excluded, and matching is based on whole tokens, so synonyms and word variants may not match.

Retrieval splits documents into blank-line-separated sections and returns up to three matching sections. If no meaningful query terms match, DocuBot returns “I don't know based on these docs.”

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

DocuBot demonstrates keyword retrieval and retrieval-augmented generation on the sample project documentation. Inputs are a question, the `.md` and `.txt` files in `docs/`, and optionally a Gemini API key for LLM-powered modes.

### Retrieval design

The retriever splits files into paragraph-sized sections and counts distinct meaningful query terms found in each section. It ranks excerpts by that count and returns up to three. Its retrieval scores are not confidence or correctness probabilities. The approach is intentionally simple and does not understand meaning, synonyms, or word variants.

### LLM behavior

Naive LLM mode does not use retrieved project evidence and may return plausible but unsupported answers. RAG mode passes the retrieved excerpts to Gemini and instructs it to rely on them and to say when the evidence is insufficient. RAG can still produce an incorrect answer, particularly when retrieval returns irrelevant or incomplete context.

### Limitations and responsible use

The sample corpus is small and illustrative. Paragraph splitting can separate a heading from the content it describes, and keyword matching can miss relevant text that uses different terminology. Check the cited source excerpts before acting on an answer, especially for security, authentication, or configuration decisions. Prefer an explicit “I don't know” to an unsupported guess.
