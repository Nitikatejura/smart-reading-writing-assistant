# Smart Reading & Writing Assistant (Classic NLP Pipeline)

A modular, end-to-end Classical Natural Language Processing pipeline that takes messy, raw paragraphs of student-submitted text (forum posts, emails, assignment answers) and processes them through five foundational computational linguistics stages.

---

## 🚀 How to Run

### 1. Installation & Environment Setup
Ensure you have Python 3.10+ installed. Install the lightweight dependencies:
```bash
pip install -r requirements.txt
```

Download the required WordNet and POS resources (performed automatically on first run, or run manually):
```bash
python -c "import nltk.downloader; nltk.downloader.download('wordnet'); nltk.downloader.download('omw-1.4'); nltk.downloader.download('averaged_perceptron_tagger_eng')"
```

### 2. Running the Pipeline via CLI
- **Run all 10 real-world messy text samples and print the evaluation summary table:**
  ```bash
  python pipeline.py --all
  ```

- **Inspect a specific test sample (e.g. Sample 1 or Sample 2) through every stage:**
  ```bash
  python pipeline.py --sample 1
  python pipeline.py --sample 2
  ```

- **Process custom raw text directly from the terminal:**
  ```bash
  python pipeline.py --text "Their is a bug in my code. Alex said he would help me. Could you please review it?"
  ```

- **Run the automated unit test suite:**
  ```bash
  python -m pytest tests/test_pipeline.py -v
  ```

### 3. Launching the Interactive Web Application (Stretch Goal: +10 pts)
Run the Streamlit interactive writing assistant:
```bash
streamlit run app.py
```
This launches a live browser interface with:
- Real-time **autocomplete-as-you-type** and instant typo detection.
- Interactive multi-tab inspection of parse trees, semantic frames, and coreference chains.
- One-click benchmark runner across all 10 student samples with comparative bar charts.

---

## 📁 Repository Structure

```text
Smart Assistant/
├── data/
│   ├── samples/                 # 10 messy real-world text samples (typos, ambiguous terms, indirect requests)
│   │   ├── sample_01.txt ... sample_10.txt
│   │   └── samples_metadata.json
│   ├── dictionary.txt           # Word frequencies for spell checking (~230k entries)
│   └── bigrams.json             # Bigram collocation frequencies for context disambiguation
├── stage1_spellcheck/           # Stage 1: Spell Checking & Homophone Disambiguation
│   ├── levenshtein.py           # From-scratch Levenshtein & Damerau DP algorithms
│   ├── ngram_context.py         # N-gram language model for context probability scoring
│   └── spell_checker.py         # Stage 1 orchestrator & diff generator
├── stage2_syntax/               # Stage 2: Syntactic Processing
│   ├── dependency_parser.py     # POS tagger and typed dependency parse tree builder
│   ├── cky_parser.py            # From-scratch CYK bottom-up dynamic programming chart parser
│   └── grammar.py               # 15 Chomsky Normal Form (CNF) CFG declarative rules
├── stage3_semantics/            # Stage 3: Semantic Analysis
│   ├── srl_extractor.py         # Syntactic dependency-based Semantic Role Labeler (Agent, Action, Patient)
│   ├── ner_tagger.py            # Named Entity Recognition (PERSON, GPE, DATE, TIME)
│   └── lesk_wsd.py              # Simplified Lesk algorithm (WordNet overlap) for ambiguous terms
├── stage4_discourse/            # Stage 4: Discourse & Pragmatic Processing
│   ├── coreference.py           # Cross-sentence anaphoric coreference resolver
│   ├── discourse_relations.py   # Connective detector & relation typing (Contrast, Cause, Elaboration, Temporal)
│   └── pragmatic_inference.py   # Speech act classifier detecting indirect requests
├── stage5_pipeline/             # Stage 5: Full Pipeline Integration
│   └── pipeline_runner.py       # Unified process() execution and sample evaluation
├── tests/
│   └── test_pipeline.py         # Automated pytest test suite (100% pass)
├── scripts/
│   └── build_language_data.py   # Language model dictionary & bigram generation script
├── pipeline.py                  # Root CLI entry point
├── app.py                       # Interactive Streamlit Web UI (Stretch Goal)
├── demo_script.md               # 5-minute video recording presentation script
└── requirements.txt             # Minimal pip requirements
```

---

## 🧠 Stage-by-Stage Demonstrations (Pedagogical Notes)

### Stage 1 — Spell Checking & Context Disambiguation
Stage 1 demonstrates non-parametric string distance algorithms by implementing a dynamic programming Levenshtein/Damerau-Levenshtein matrix from scratch without third-party libraries. It generates candidate corrections at edit distance $\le 2$ and ranks them using unigram vocabulary frequencies. Crucially, it incorporates a context-aware bigram language model with Laplace smoothing to resolve ambiguous homophones (e.g., *there* vs *their*, *to* vs *two*) that edit distance alone cannot differentiate. The output yields both the corrected text and a detailed change log explaining the exact edit distance or log-probability score driving every replacement.

### Stage 2 — Syntactic Processing
Stage 2 demonstrates hierarchical sentence structure analysis through both statistical dependency parsing and formal grammar chart parsing. It assigns Penn Treebank Part-of-Speech tags and produces typed dependency parse trees (capturing `ROOT`, `nsubj`, `dobj`, `prep`, `pobj`, and modifiers) rendered in clean ASCII hierarchies. Additionally, it implements a from-scratch Cocke-Younger-Kasami (CYK) dynamic programming algorithm over a hand-crafted Chomsky Normal Form (CNF) grammar of 15 rules. This proves direct mastery of algorithmic chart parsing mechanics, non-terminal parent reduction, and bracketed phrase-structure tree derivation.

### Stage 3 — Semantic Analysis
Stage 3 demonstrates predicate-argument extraction and lexical ambiguity resolution. Using syntactic dependency patterns over the parse trees, it extracts semantic roles—identifying the primary predicate (Action), nominal subject (Agent / Arg0), direct object (Patient / Arg1), and prepositional modifiers. It integrates Named Entity Recognition to detect real-world entities (PERSON, GPE, DATE, TIME) across student communications. Finally, it implements the Simplified Lesk algorithm from scratch, computing token overlaps between surrounding context and WordNet glosses/examples to disambiguate polysemous words (e.g., distinguishing a river *bank* from a financial *bank*).

### Stage 4 — Discourse & Pragmatic Processing
Stage 4 elevates analysis beyond isolated sentences to paragraph-level discourse and pragmatic intent. It features a heuristic coreference resolver that tracks entity salience and resolves third-person anaphoric pronouns (*he*, *she*, *it*, *they*) to their proper cross-sentence antecedents using gender/number agreement and recency. It scans for explicit discourse connectives (e.g., *however*, *therefore*, *because*, *furthermore*) and categorizes the rhetorical relations they signal (Contrast, Cause/Result, Elaboration, Temporal). Furthermore, it implements pragmatic speech act inference to detect indirect requests (e.g., recognizing *"Could you please grant me an extension?"* as a directive request rather than an inquiry into the listener's physical ability).

### Stage 5 — Full Pipeline Assembly
Stage 5 assembles all four computational linguistics stages into a unified `process(raw_text)` function that flows from raw paragraph to fully structured linguistic representations. It evaluates a curated corpus of 10 real-world messy student texts (forum posts, late-submission excuse emails, peer reviews, lab inquiries) containing authentic typos, pronoun ambiguity, and rhetorical markers. The pipeline automatically tabulates aggregate metrics per sample, quantifying spelling corrections, recognized entities, resolved coreference chains, discourse relations, and pragmatic speech acts.

---

## 📊 Benchmark Summary Table (10 Real-World Test Samples)

The pipeline was executed across all 10 messy student samples in `data/samples/`, yielding the following comparative evaluation metrics:

| Sample File | Title / Domain | # Spelling Corrections | # Entities Found | # Coref Chains | # Discourse Relations | # Pragmatic Inferences |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `sample_01.txt` | Assignment Extension Request | 2 | 2 | 2 | 3 | 1 |
| `sample_02.txt` | Essay Draft Consultation (*bank* WSD) | 1 | 4 | 1 | 3 | 1 |
| `sample_03.txt` | Sports Biology Doubt (*bat/light* WSD) | 0 | 3 | 1 | 1 | 1 |
| `sample_04.txt` | Botanical Experiment Log (*plant* WSD) | 1 | 2 | 2 | 1 | 1 |
| `sample_05.txt` | Group Project Bug Report | 2 | 6 | 1 | 2 | 1 |
| `sample_06.txt` | Astronomy Research Note (*star* WSD) | 0 | 0 | 1 | 2 | 1 |
| `sample_07.txt` | Economics Peer Review (*bank* WSD) | 1 | 1 | 0 | 2 | 1 |
| `sample_08.txt` | Solar Energy Lab Note | 1 | 4 | 1 | 2 | 1 |
| `sample_09.txt` | Hackathon Team Update | 0 | 2 | 1 | 2 | 1 |
| `sample_10.txt` | Library Research Query | 0 | 3 | 2 | 1 | 1 |

---

## 🌟 Stretch Goal Feature: Autocomplete & Assistant UI (+10 pts)
The Streamlit app (`app.py`) transforms the command-line script into an interactive web-based writing assistant:
1. **Autocomplete-as-you-type**: Continuously monitors the active cursor word and displays real-time vocabulary completions ranked by corpus frequency.
2. **Instant Typo & Grammar Diff**: Shows side-by-side color-coded badges highlighting misspelled tokens and context homophone mistakes with their dynamic programming edit distances.
3. **Speech Act Guidance**: Prompts the writer when their interrogative question is functioning pragmatically as an action request.
