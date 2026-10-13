# 5-Minute Demo Video Recording Script
## Smart Reading & Writing Assistant (Classic NLP Pipeline)

Use this time-stamped script to record a crisp, high-scoring 5-minute demonstration video of the pipeline running live on real-world text samples.

---

### [0:00 – 0:45] Minute 1: Introduction & Architecture Overview
- **What to show on screen**: Open VS Code or terminal showing the folder structure: `stage1_spellcheck/`, `stage2_syntax/`, `stage3_semantics/`, `stage4_discourse/`, `stage5_pipeline/`, and `pipeline.py`.
- **Spoken Narration**:
  > *"Hello! Today I am presenting our Smart Reading & Writing Assistant, an end-to-end Classical NLP pipeline built to assist students with real-world, messy written text like forum questions, late assignment emails, and peer feedback.*  
  > *Instead of isolated exercises, every topic from the syllabus is connected into five sequential stages: from-scratch Levenshtein spell checking with context N-grams, syntactic dependency parsing with a from-scratch CYK parser, semantic role labeling and Lesk Word-Sense Disambiguation, cross-sentence coreference with pragmatic speech act inference, and full pipeline assembly evaluated on 10 authentic text samples."*

---

### [0:45 – 2:00] Minute 2: Live Deep Dive on Sample 1 (Assignment Extension)
- **What to do**: In terminal, run:
  ```bash
  python pipeline.py --sample 1
  ```
- **What to highlight as output scrolls**:
  1. **Stage 1 (Spell Check)**: Point out:
     > *"Notice in Sample 1 the homophone typo 'Their is a major bug'. Because 'Their' is a real dictionary word, basic edit distance fails. Our context-aware bigram language model evaluates P('there is') vs P('their is') and correctly chooses 'There'. Simultaneously, our from-scratch Levenshtein matrix corrects 'tonite' to 'tonight' with an edit distance of 3."*
  2. **Stage 2 (Syntax)**: Point out the ASCII dependency tree:
     > *"In Stage 2, our dependency parser assigns POS tags and draws the full hierarchical tree, identifying the root predicate, nominal subjects, direct objects, and prepositional modifiers. We also demonstrate our from-scratch CYK chart parser parsing a declarative sentence over our 15-rule CNF grammar."*
  3. **Stage 3 (Semantics)**: Point out Agent / Action / Patient:
     > *"In Stage 3, we extract semantic frames: 'Alex' is the Agent, 'said' is the Action, and 'he' is the Patient, alongside Named Entity Recognition tagging 'Alex' as PERSON and 'tonight' as DATE."*
  4. **Stage 4 (Discourse & Pragmatics)**: Point out:
     > *"In Stage 4, cross-sentence coreference links the pronoun 'he' and possessive 'his' back to 'Alex'. Connectives 'but' and 'however' are labeled as Contrast relations. Most importantly, our pragmatic speech act rule inspects 'Could you please grant me an extension on the assignment?' and correctly infers that it is an indirect directive request, not a literal yes/no question."*

---

### [2:00 – 3:00] Minute 3: Live Deep Dive on Sample 2 (River Bank vs Financial Bank WSD)
- **What to do**: In terminal, run:
  ```bash
  python pipeline.py --sample 2
  ```
- **What to highlight**:
  1. **Stage 1 (Transposition)**:
     > *"In Sample 2, the typo 'recieved' has transposed letters. Our Damerau-Levenshtein algorithm handles adjacent transpositions as a single edit step, accurately restoring 'received'."*
  2. **Stage 3 (Word-Sense Disambiguation via Lesk)**:
     > *"Look closely at the polysemous word 'bank' in 'river bank ecosystem'. Our Simplified Lesk algorithm compares the surrounding sentence context against all WordNet synsets. Because of the token overlap with 'river', it selects `bank.n.01`—sloping land beside water—rather than a financial institution."*
  3. **Stage 4 (Pragmatic Request & Cause Connective)**:
     > *"Connectives 'Therefore' and 'but' are tagged for Cause and Contrast, and 'Can you please review the draft before Thursday?' is flagged as an indirect action request."*

---

### [3:00 – 4:15] Minute 4: Stretch Goal — Interactive Streamlit Assistant (+10 pts)
- **What to do**: Run and switch to browser:
  ```bash
  streamlit run app.py
  ```
- **What to demonstrate on screen**:
  1. **Live Autocomplete**:
     > *"For our stretch goal, we built an interactive Streamlit writing assistant. As I type in the editor, our prefix-matching frequency tree instantly generates real-time word completions that can be inserted with one click."*
  2. **Live Grammar & Spell Checking**:
     > *"As text is entered, typos like 'recieved' are highlighted in real-time with red badges, displaying the exact dynamic programming edit distance and rationale."*
  3. **Multi-Stage Explorer**:
     > *"Users can switch to the 5-Stage Interactive Pipeline tab, pick any of our 10 preset samples, and visually inspect dependency trees, semantic frames, and coreference chains."*
  4. **10-Sample Benchmark Tab**:
     > *"Click 'Run Full 10-Sample Benchmark' to view the automated evaluation matrix and comparative bar chart across all 10 student samples."*

---

### [4:15 – 5:00] Minute 5: Automated Verification & Conclusion
- **What to do**: Switch back to terminal and run the test suite:
  ```bash
  python -m pytest tests/test_pipeline.py -v
  ```
- **What to highlight**:
  > *"All 9 unit tests across all 5 stages pass with 100% success—verifying Levenshtein DP calculation, N-gram context scoring, CYK chart parsing, dependency trees, Lesk WSD, semantic roles, coreference resolution, and pragmatic speech act detection.*  
  > *Thank you! The entire codebase, test suite, and documentation are ready in the repository."*
