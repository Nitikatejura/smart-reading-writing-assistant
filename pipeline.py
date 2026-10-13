"""
Root CLI Entry Point for the Smart Reading & Writing Assistant (NLP Pipeline).

Usage:
  python pipeline.py --all            # Run all 10 test samples and output summary table
  python pipeline.py --sample 1       # Run detailed multi-stage inspection on sample 01
  python pipeline.py --text "..."     # Process a custom text string
"""
import sys
import argparse
import json
import os

from stage5_pipeline.pipeline_runner import SmartAssistantPipeline, PipelineResult
from stage2_syntax.cky_parser import CYKParser

_shared_pipeline = None

def process(raw_text: str) -> PipelineResult:
    """
    Wired stages 1->4 into a single function as specified in Stage 5:
    process(raw_text) -> corrected_text, parse_trees, semantic_frames, discourse_relations, coref_chains
    
    Supports both direct tuple unpacking:
      corrected_text, parse_trees, semantic_frames, discourse_relations, coref_chains = process(raw_text)
    and attribute access:
      res = process(raw_text)
      print(res.corrected_text)
    """
    global _shared_pipeline
    if _shared_pipeline is None:
        _shared_pipeline = SmartAssistantPipeline()
    return _shared_pipeline.process(raw_text)


def run_single_sample_inspection(pipeline: SmartAssistantPipeline, sample_num: int, samples_dir: str = None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"sample_{sample_num:02d}.txt"
    if samples_dir:
        filepath = os.path.join(samples_dir, filename)
    else:
        # Check test_samples/ first, then data/samples/
        cand1 = os.path.join(base_dir, "test_samples", filename)
        cand2 = os.path.join(base_dir, "data", "samples", filename)
        filepath = cand1 if os.path.exists(cand1) else cand2

    if not os.path.exists(filepath):
        print(f"Error: Sample file not found at {filepath}")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("=" * 80)
    print(f"PIPELINE INSPECTION: {filename}")
    print("=" * 80)
    print(f"\n[RAW INPUT TEXT]:\n{raw_text.strip()}\n")

    res = pipeline.process(raw_text)

    # Stage 1
    print("-" * 80)
    print("STAGE 1: SPELL CHECKING & CONTEXT DISAMBIGUATION")
    print("-" * 80)
    print(f"Corrected Text:\n{res.corrected_text.strip()}\n")
    if res.spelling_diffs:
        print(f"Found {len(res.spelling_diffs)} correction(s):")
        for d in res.spelling_diffs:
            print(f"  * '{d['original']}' -> '{d['corrected']}' [edit_distance: {d.get('edit_distance', 'N/A')}]")
            print(f"    Reason: {d['reason']}")
    else:
        print("No spelling errors detected.")

    # Stage 2
    print("\n" + "-" * 80)
    print("STAGE 2: SYNTACTIC PROCESSING (POS & DEPENDENCY TREES)")
    print("-" * 80)
    for idx, pt in enumerate(res.parse_trees, 1):
        print(f"\nSentence {idx}: {pt['sentence']}")
        print(f"POS Sequence: {pt['pos_tags']}")
        print(f"Dependency Parse Tree:\n{pt['ascii_tree']}")

    # CKY Demo
    print("\n--- Hand-Crafted CYK Parser Demo on Declarative Grammar ---")
    cyk = pipeline.cyk_parser
    demo_sentence = "The astronomer pointed the telescope"
    cyk_res = cyk.parse(demo_sentence)
    if cyk_res["success"]:
        print(f"Input: '{demo_sentence}'")
        print(f"Bracket Tree: {cyk_res['bracket_tree']}")
        print(f"ASCII Tree:\n{cyk_res['ascii_tree']}")

    # Stage 3
    print("\n" + "-" * 80)
    print("STAGE 3: SEMANTIC ANALYSIS (SRL, NER & LESK WSD)")
    print("-" * 80)
    for idx, sf in enumerate(res.semantic_frames, 1):
        print(f"\nSentence {idx}: {sf['sentence']}")
        print(f"  Agent   : {sf['agent']}")
        print(f"  Action  : {sf['action']}")
        print(f"  Patient : {sf['patient']}")
        if sf.get("modifiers"):
            print(f"  Modifier: {sf['modifiers']}")
        if sf.get("entities"):
            ent_strs = [e["text"] + " (" + e["label"] + ")" for e in sf["entities"]]
            print(f"  Entities: {ent_strs}")
        if sf.get("wsd"):
            for w in sf["wsd"]:
                print(f"  WSD [{w['word']}]: Synset '{w['selected_synset']}' ({w['definition']}) | Overlap: {w['overlap_words']}")

    # Stage 4
    print("\n" + "-" * 80)
    print("STAGE 4: DISCOURSE & PRAGMATIC PROCESSING")
    print("-" * 80)
    print("Coreference Chains:")
    if res.coref_chains:
        for c in res.coref_chains:
            print(f"  * Antecedent '{c['antecedent']}' ({c.get('category', 'ENTITY')}): mentions = {c['mentions']}")
    else:
        print("  None detected.")

    print("\nDiscourse Relations:")
    if res.discourse_relations:
        for dr in res.discourse_relations:
            print(f"  * [{dr['relation_type']}] Connective '{dr['connective']}' (Sentence {dr['sentence_idx']})")
            print(f"    Arg1: {dr['arg1']}")
            print(f"    Arg2: {dr['arg2']}")
    else:
        print("  None detected.")

    print("\nPragmatic Inferences:")
    if res.pragmatic_inferences:
        for pi in res.pragmatic_inferences:
            print(f"  * Sentence: '{pi['sentence']}'")
            print(f"    Surface Form: {pi['surface_form']}")
            print(f"    Pragmatic Speech Act: {pi['pragmatic_force']}")
            print(f"    Interpretation: {pi['interpretation']}")
            print(f"    Note: {pi['explanation']}")
    else:
        print("  None detected.")

    # Stage 5
    print("\n" + "=" * 80)
    print("STAGE 5: SUMMARY COUNTS FOR THIS SAMPLE")
    print("=" * 80)
    for k, v in res.summary_counts.items():
        print(f"  {k:30s}: {v}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Smart Reading & Writing Assistant (NLP Pipeline)")
    parser.add_argument("--all", action="store_true", help="Run full pipeline evaluation across all 10 test samples")
    parser.add_argument("--sample", type=int, default=None, help="Inspect a specific sample number (1-10) in depth")
    parser.add_argument("--samples-dir", type=str, default=None, help="Custom folder containing test sample text files")
    parser.add_argument("--text", type=str, default=None, help="Process custom raw input text")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()
    pipeline = SmartAssistantPipeline()

    # Determine default samples directory: test_samples/ or data/samples/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = args.samples_dir
    if not samples_dir:
        cand1 = os.path.join(base_dir, "test_samples")
        cand2 = os.path.join(base_dir, "data", "samples")
        samples_dir = cand1 if os.path.exists(cand1) else cand2

    if args.sample is not None:
        run_single_sample_inspection(pipeline, args.sample, samples_dir=samples_dir)
    elif args.text is not None:
        res = pipeline.process(args.text)
        if args.json:
            print(json.dumps(res.to_dict(), indent=2))
        else:
            print("\nCorrected Text:\n", res.corrected_text)
            print("\nSummary Counts:\n", json.dumps(res.summary_counts, indent=2))
    else:
        # Default or --all: run all 10 samples and display table
        eval_results = pipeline.evaluate_samples(samples_dir=samples_dir)
        table = pipeline.format_summary_table(eval_results)
        print("\n" + "=" * 80)
        print("STAGE 5: FULL PIPELINE EVALUATION SUMMARY TABLE (10 SAMPLES)")
        print("=" * 80)
        print(table)
        print("=" * 80)


if __name__ == "__main__":
    main()
