# DocuBot

DocuBot is a small documentation assistant that helps answer developer questions about a codebase. It can operate in three different modes:

1. **Naive LLM mode**  
   Sends the entire documentation corpus to a Gemini model and asks it to answer the question.

2. **Retrieval only mode**  
   Uses a simple indexing and scoring system to retrieve relevant snippets without calling an LLM.

3. **RAG mode (Retrieval Augmented Generation)**  
   Retrieves relevant snippets, then asks Gemini to answer using only those snippets.

The docs folder contains realistic developer documents (API reference, authentication notes, database notes), but these files are **just text**. They support retrieval experiments and do not require students to set up any backend systems.

---

## App Structure

'''
DocuBot/
│
├── docs/
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
'''

---

## Setup

### 1. Install Python dependencies

    pip install -r requirements.txt

### 2. Configure environment variables

Copy the example file:

    cp .env.example .env

Then edit `.env` to include your Gemini API key:

    GEMINI_API_KEY=your_api_key_here

If you do not set a Gemini key, you can still run retrieval only mode.

---

## Running DocuBot

Start the command-line program:

    python main.py

To use the web chat interface:

    streamlit run app.py

The web interface supports Naive LLM, Retrieval only, and RAG modes. It shows retrieved source sections, keeps a conversation history with a clear button, and can compare Retrieval only with RAG on the same question. Gemini modes and comparison require `GEMINI_API_KEY`. Example supported and unsupported questions are included in the app.

Choose a mode:

- **1**: Naive LLM (Gemini reads the full docs)  
- **2**: Retrieval only (no LLM)  
- **3**: RAG (retrieval + Gemini)

You can use built in sample queries or type your own.

---

## Running Retrieval Evaluation (optional)

    python evaluation.py

This prints simple retrieval hit rates for sample queries.

---

## Modifying the Project

You will primarily work in:

- `docubot.py`  
  Implement or improve the retrieval index, scoring, and snippet selection.

- `llm_client.py`  
  Adjust the prompts and behavior of LLM responses.

- `dataset.py`  
  Add or change sample queries for testing.

---

## Requirements

- Python 3.9+
- A Gemini API key for LLM features (only needed for modes 1 and 3)
- No database, no server setup, no external services besides LLM calls

---

## Model Card (I will delete some of it later)

Short reflection on the DocuBot system

---

### System Overview

**What is DocuBot trying to do?**
DocuBot answers developer questions about a project by searching its documentation (`docs/`) and, optionally, using an LLM to phrase an answer grounded in what it finds. It exists to show the difference between generating from training data and generating from retrieved evidence.

**What inputs does DocuBot take?** A user question, the Markdown/text files in `docs/`, and (for Mode 3) a `GEMINI_API_KEY` in `.env`.

**What outputs does DocuBot produce?** Either raw retrieved snippets with their source filenames (Mode 2), an LLM answer grounded in those snippets (Mode 3), or an ungrounded LLM answer (Mode 1). When nothing relevant is found, it returns "I do not know based on these docs."

---

### Retrieval Design

**How does retrieval work?**
- **Index:** `build_index` builds an inverted index (token → list of files).
- **Chunking:** each document is split into paragraph-sized chunks (`_build_chunks`), so retrieval can return one focused section instead of a whole file.
- **Scoring:** `score_document` drops stopwords (including corpus-meta words like "docs") from the query, then counts how many remaining query words appear as whole tokens in a chunk.
- **Selection:** `retrieve` scores every chunk, sorts descending, and returns the top 3 — unless the best chunk scores below the confidence threshold, in which case it returns nothing (a refusal).

**Tradeoffs:** Whole-token matching is precise but misses word variants ("connect" vs "connection", "database" vs "database_url"). Paragraph chunks improve focus but can split a heading away from the table it introduces. Simplicity was chosen over accuracy on purpose.

---

### Use of the LLM (Gemini)

- **Naive LLM mode:** answers from training data only, ignoring the docs — a pure
  hallucination baseline.
- **Retrieval only mode:** no LLM; returns the retrieved snippets verbatim.
- **RAG mode:** retrieves snippets first, then asks Gemini to answer using only those snippets and to refuse when they're insufficient.

**Grounding instructions:** the RAG prompt in `llm_client.py` tells the model to use only the provided snippets and to say "I do not know" rather than guess.

---

### Experiments and Comparisons

Same queries, all three modes (Modes 1/3 require a Gemini key; observations below are from Mode 2 retrieval plus expected LLM behavior):

| Query | Naive LLM | Retrieval only | RAG | Notes |
|------|-----------|----------------|-----|-------|
| Where is the auth token generated? | Plausible but unverifiable | Returns the AUTH.md token section | Grounded + readable | Retrieval nails this one |
| How do I connect to the database? | Generic advice | Returns SETUP.md, not DATABASE.md | Only as good as retrieval | Whole-token miss: "database_url" ≠ "database" |
| Which endpoint lists all users? | May invent a path | Returns the correct API_REFERENCE.md route | Best of the three | Clear RAG win |
| Is there any mention of payment processing? | Might hallucinate a feature | **Refuses** | **Refuses** | Guardrail working as intended |

**Patterns:** Mode 1 is fluent but ungrounded — dangerous when it invents specifics. Mode 2 is trustworthy but raw. Mode 3 is best when retrieval is right, and *inherits retrieval's mistakes* when it's wrong.

---

### Failure Cases and Guardrails

**Failure case 1 — "How do I connect to the database?"** Retrieval returns SETUP.md instead of DATABASE.md, because the DATABASE doc says "connection" and "DATABASE_URL" — neither is a whole-token match for "connect"/"database". A stemming or substring match would fix it.

**Failure case 2 — "How does a client refresh an access token?"** Retrieval returns API_REFERENCE.md (which documents `POST /api/refresh`) rather than AUTH.md's refresh narrative. Both are arguably relevant; the ranking just favors the endpoint doc.

**When DocuBot should refuse:** (1) when the question is about something not in the docs at all (payment processing); (2) when no chunk shares a meaningful keyword with the question.

**Guardrail:** if the top-scoring chunk doesn't clear the confidence threshold, `retrieve` returns an empty list and every answer mode replies "I do not know based on these docs."

---

### Limitations and Future Improvements

**Limitations**
1. Keyword matching has no understanding of synonyms or word variants.
2. Paragraph chunking can separate a heading from the content it labels.
3. The corpus is tiny and fictional; scores don't reflect real-world scale.

**Future improvements**
1. Add stemming/lemmatization or substring matching to catch word variants.
2. Weight by term frequency (TF-IDF) instead of presence counts.
3. Rank by document-level relevance before selecting chunks, to avoid one verbose file crowding the top results.

---

### Responsible Use

**Where could this cause harm?** A confident but wrong answer about auth or database config could send a developer down the wrong path; a naive-mode hallucination could invent an endpoint or env var that doesn't exist.

**Guidelines for real developers:**
- Treat Mode 1 (no retrieval) answers as unverified guesses.
- Always check the cited source file before acting on an answer.
- Prefer a clear "I do not know" over a plausible guess — the guardrail exists for a reason.
