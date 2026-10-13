"""
Streamlit Web Application: Smart Reading & Writing Assistant (NLP Pipeline).
Implements the interactive Writing Assistant with live autocomplete and full 5-stage inspection.
"""
import os
import glob
import json
import streamlit as st

from stage1_spellcheck.spell_checker import SpellChecker
from stage1_spellcheck.levenshtein import CandidateGenerator, damerau_levenshtein_distance
from stage2_syntax.dependency_parser import DependencyParser
from stage2_syntax.cky_parser import CYKParser
from stage3_semantics.srl_extractor import SRLExtractor
from stage3_semantics.lesk_wsd import LeskWSD
from stage4_discourse.coreference import CoreferenceResolver
from stage4_discourse.discourse_relations import DiscourseRelationDetector
from stage4_discourse.pragmatic_inference import PragmaticInferenceAnalyzer
from stage5_pipeline.pipeline_runner import SmartAssistantPipeline


st.set_page_config(
    page_title="Smart NLP Reading & Writing Assistant",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_pipeline():
    return SmartAssistantPipeline()


pipeline = load_pipeline()

# Header
st.title("✍️ Smart Reading & Writing Assistant")
st.markdown("""
An end-to-end Classical NLP system processing text across five foundational stages:
**Spell Checking** (from-scratch Levenshtein & N-grams) $\\rightarrow$ **Syntactic Processing** (Dependency & from-scratch CYK) $\\rightarrow$
**Semantic Analysis** (SRL, NER, Lesk WSD) $\\rightarrow$ **Discourse & Pragmatics** (Coreference, Connectives, Speech Acts) $\\rightarrow$
**Pipeline Synthesis**.
""")

# Sidebar: Preset Sample Selector
st.sidebar.header("📁 Real-World Messy Text Samples")
samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples")
sample_files = sorted(glob.glob(os.path.join(samples_dir, "sample_*.txt")))

sample_options = ["Custom Text"] + [os.path.basename(f) for f in sample_files]
selected_sample = st.sidebar.selectbox("Choose a test sample:", sample_options)

default_text = "Hi Professor, I hope you are doing well. Their is a major bug in my code that I cannot fix before tonite. Alex said he would help me with it, but he is busy with his own project. Could you please grant me an extension on the assignment? However, I will submit what I have so far if necessary."

if selected_sample != "Custom Text":
    sample_path = os.path.join(samples_dir, selected_sample)
    if os.path.exists(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            default_text = f.read()

# Main tabs
tab_write, tab_pipeline, tab_batch = st.tabs([
    "⚡ Stretch Goal: Autocomplete & Live Assistant",
    "🔬 5-Stage Interactive NLP Pipeline",
    "📊 10-Sample Evaluation Benchmark"
])

# ==============================================================================
# TAB 1: Autocomplete & Live Spell Checker (Stretch Goal)
# ==============================================================================
with tab_write:
    st.subheader("⚡ Live Writing Assistant & Autocomplete Simulator")
    st.caption("Demonstrating real-time prefix-matching suggestions and from-scratch Levenshtein spelling corrections as you type.")

    col1, col2 = st.columns([2, 1])

    with col1:
        current_input = st.text_area(
            "Type your text here:",
            value="Dear Dr. Watson, I wanted to discuss my essay about the river bank ecosystem. It has recieved good remarks, but some points require clarity. Can you please review the draft before Thursday?",
            height=140
        )

        # Autocomplete on the last word being typed
        last_word = ""
        tokens = current_input.split()
        if tokens:
            last_word = tokens[-1].strip(".,!?;:\"'").lower()

        # Find prefix completions
        completions = []
        if last_word and len(last_word) >= 2:
            vocab = pipeline.spell_checker.vocab
            matched = [w for w in vocab.keys() if w.startswith(last_word) and w != last_word]
            completions = sorted(matched, key=lambda w: -vocab.get(w, 0))[:6]

        st.markdown("**🔮 Autocomplete Suggestions for current word:**")
        if completions:
            cols = st.columns(len(completions))
            for i, comp in enumerate(completions):
                cols[i].button(comp, key=f"comp_{i}")
        else:
            st.info("Type 2 or more letters to see vocabulary completions...")

    with col2:
        st.markdown("**🔍 Live Spelling & Grammar Check:**")
        if current_input.strip():
            spell_res = pipeline.spell_checker.check(current_input)
            if spell_res["diffs"]:
                st.warning(f"Found {len(spell_res['diffs'])} suggested correction(s):")
                for d in spell_res["diffs"]:
                    with st.container():
                        st.markdown(f"❌ **`{d['original']}`** $\\rightarrow$ ✅ **`{d['corrected']}`**")
                        st.caption(f"Reason: {d['reason']}")
            else:
                st.success("✅ No spelling errors detected!")

            # Pragmatic check
            prag = pipeline.pragmatic_analyzer.analyze_paragraph(current_input)
            if prag:
                st.info(f"💡 **Speech Act Detected**: {prag[0]['pragmatic_force']}\n\n*{prag[0]['interpretation']}*")

# ==============================================================================
# TAB 2: Full 5-Stage Interactive Pipeline
# ==============================================================================
with tab_pipeline:
    st.subheader("🔬 End-to-End Pipeline Analysis")
    user_text = st.text_area("Input paragraph to process through the pipeline:", value=default_text, height=130)

    if st.button("🚀 Process Through Full NLP Pipeline", type="primary"):
        with st.spinner("Executing Stages 1 through 4..."):
            res = pipeline.process(user_text)

        st.success("Pipeline execution complete!")

        # Metrics bar
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Spelling Corrections", res.summary_counts["num_spelling_corrections"])
        m2.metric("Entities Found", res.summary_counts["num_entities"])
        m3.metric("Coref Chains", res.summary_counts["num_coref_chains"])
        m4.metric("Discourse Relations", res.summary_counts["num_discourse_relations"])
        m5.metric("Pragmatic Inferences", res.summary_counts["num_pragmatic_inferences"])

        stage1_tab, stage2_tab, stage3_tab, stage4_tab = st.tabs([
            "Stage 1: Spell Check",
            "Stage 2: Syntax & Parsing",
            "Stage 3: Semantics & WSD",
            "Stage 4: Discourse & Pragmatics"
        ])

        with stage1_tab:
            st.markdown("### Stage 1: Spell Checking & Context Disambiguation")
            st.markdown("**Original Text:**")
            st.info(res.raw_text)
            st.markdown("**Corrected Text:**")
            st.success(res.corrected_text)

            if res.spelling_diffs:
                st.markdown("#### Replacement Diff Log")
                diff_data = []
                for d in res.spelling_diffs:
                    diff_data.append({
                        "Original": d["original"],
                        "Corrected": d["corrected"],
                        "Edit Distance": d.get("edit_distance", "N/A"),
                        "Rationale": d["reason"]
                    })
                st.table(diff_data)
            else:
                st.info("No words were changed in Stage 1.")

        with stage2_tab:
            st.markdown("### Stage 2: Syntactic Processing")
            st.markdown("#### A. Dependency Parse Trees & POS Tags")
            for idx, pt in enumerate(res.parse_trees, 1):
                with st.expander(f"Sentence {idx}: {pt['sentence']}", expanded=(idx == 1)):
                    st.markdown("**Part-of-Speech (POS) Tags:**")
                    st.code(" ".join([f"{word}/{tag}" for word, tag in pt["pos_tags"]]))
                    st.markdown("**ASCII Dependency Tree:**")
                    st.code(pt["ascii_tree"])

            st.markdown("#### B. From-Scratch CYK (CKY) Chart Parser Demo")
            st.caption("Bottom-up dynamic programming parser verifying Chomsky Normal Form (CNF) grammar mechanics.")
            cnf_sentence = st.text_input("Test CYK Parser with a simple declarative sentence:", value="The astronomer pointed the telescope")
            if st.button("Run CYK Parser"):
                cyk_res = pipeline.cyk_parser.parse(cnf_sentence)
                if cyk_res["success"]:
                    st.success(f"Grammar accepted! Found {cyk_res['num_parses']} parse tree(s).")
                    st.code(cyk_res["bracket_tree"], language="text")
                    st.code(cyk_res["ascii_tree"], language="text")
                else:
                    st.warning("Sentence could not be parsed with current 15-rule CNF grammar.")

        with stage3_tab:
            st.markdown("### Stage 3: Semantic Analysis")
            st.markdown("#### A. Semantic Role Labeling (SRL) Frames")
            srl_rows = []
            for idx, sf in enumerate(res.semantic_frames, 1):
                srl_rows.append({
                    "Sentence": f"S{idx}: {sf['sentence'][:40]}...",
                    "Agent (Who)": sf["agent"],
                    "Action (Predicate)": sf["action"],
                    "Patient (Whom/What)": sf["patient"],
                    "Modifiers": sf["modifiers"] or "-"
                })
            st.table(srl_rows)

            st.markdown("#### B. Named Entity Recognition (NER)")
            all_ents = []
            for idx, sf in enumerate(res.semantic_frames, 1):
                for e in sf.get("entities", []):
                    all_ents.append({
                        "Sentence": idx,
                        "Entity Text": e["text"],
                        "Entity Label": e["label"]
                    })
            if all_ents:
                st.dataframe(all_ents, use_container_width=True)
            else:
                st.info("No named entities detected.")

            st.markdown("#### C. Word-Sense Disambiguation (Simplified Lesk Algorithm)")
            st.caption("Computes WordNet gloss, example, and hypernym token overlaps to pick the contextually correct word sense.")
            wsd_found = False
            for sf in res.semantic_frames:
                for w in sf.get("wsd", []):
                    wsd_found = True
                    st.markdown(f"**Target Word:** `{w['word']}` $\\rightarrow$ **Selected Synset:** `{w['selected_synset']}` (Overlap Score: {w['score']})")
                    st.markdown(f"*Definition:* {w['definition']}")
                    st.markdown(f"*Context Overlap Words:* `{w['overlap_words']}`")
                    with st.expander(f"View candidate senses for '{w['word']}'"):
                        st.json(w["all_candidates"])
            if not wsd_found:
                st.info("No polysemous target words (bank, bat, light, plant, star) detected in this sample.")

        with stage4_tab:
            st.markdown("### Stage 4: Discourse & Pragmatic Processing")
            st.markdown("#### A. Cross-Sentence Coreference Resolution")
            if res.coref_chains:
                for c in res.coref_chains:
                    st.markdown(f"🔗 **Antecedent:** `{c['antecedent']}` ({c.get('category', 'ENTITY')})")
                    st.markdown(f"**Mentions & Anaphors:** {c['mentions']}")
            else:
                st.info("No coreference chains resolved in this sample.")

            st.markdown("#### B. Discourse Connectives & Relation Typing")
            if res.discourse_relations:
                dr_data = []
                for dr in res.discourse_relations:
                    dr_data.append({
                        "Connective": dr["connective"],
                        "Relation Type": dr["relation_type"],
                        "Sentence": dr["sentence_idx"],
                        "Arg 1": dr["arg1"],
                        "Arg 2": dr["arg2"]
                    })
                st.table(dr_data)
            else:
                st.info("No discourse connectives detected.")

            st.markdown("#### C. Pragmatic Inference (Indirect Speech Act Detection)")
            if res.pragmatic_inferences:
                for pi in res.pragmatic_inferences:
                    st.warning(f"💡 **Speech Act Detected:** {pi['pragmatic_force']}")
                    st.markdown(f"**Sentence:** *\"{pi['sentence']}\"*")
                    st.markdown(f"**Surface Form:** {pi['surface_form']}")
                    st.markdown(f"**Pragmatic Interpretation:** **{pi['interpretation']}**")
                    st.caption(f"Rule: {pi['explanation']}")
            else:
                st.info("No indirect speech acts identified in this sample.")

# ==============================================================================
# TAB 3: Batch 10-Sample Benchmark
# ==============================================================================
with tab_batch:
    st.subheader("📊 10 Real-World Messy Text Samples Benchmark")
    st.markdown("Run all 10 student-submitted texts through the pipeline to generate the full comparative evaluation matrix.")

    if st.button("⚡ Run Full 10-Sample Benchmark", type="secondary"):
        with st.spinner("Processing all 10 samples..."):
            eval_results = pipeline.evaluate_samples()

        summary_rows = []
        for r in eval_results:
            summary_rows.append({
                "Sample": r["sample"],
                "Spelling Corrections": r["corrections"],
                "Entities Found": r["entities"],
                "Coref Chains": r["coref_chains"],
                "Discourse Relations": r["discourse_relations"],
                "Pragmatic Inferences": r["pragmatic_inferences"]
            })
        st.dataframe(summary_rows, use_container_width=True)

        # Bar chart
        try:
            import pandas as pd
            df = pd.DataFrame(summary_rows).set_index("Sample")
            st.bar_chart(df)
        except Exception:
            pass
