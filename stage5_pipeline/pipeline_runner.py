"""
Stage 5: Unified Pipeline Runner.
Connects Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 into a cohesive writing assistant pipeline.
"""
import os
import glob
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from stage1_spellcheck.spell_checker import SpellChecker
from stage2_syntax.dependency_parser import DependencyParser
from stage2_syntax.cky_parser import CYKParser
from stage3_semantics.srl_extractor import SRLExtractor
from stage3_semantics.lesk_wsd import LeskWSD
from stage4_discourse.coreference import CoreferenceResolver
from stage4_discourse.discourse_relations import DiscourseRelationDetector
from stage4_discourse.pragmatic_inference import PragmaticInferenceAnalyzer


@dataclass
class PipelineResult:
    """Structured container holding results from all 5 pipeline stages."""
    raw_text: str
    corrected_text: str
    spelling_diffs: List[Dict[str, Any]]
    parse_trees: List[Dict[str, Any]]
    semantic_frames: List[Dict[str, Any]]
    discourse_relations: List[Dict[str, Any]]
    coref_chains: List[Dict[str, Any]]
    pragmatic_inferences: List[Dict[str, Any]]
    summary_counts: Dict[str, int]

    def __iter__(self):
        """Allows direct tuple unpacking as specified in the assignment:
        corrected_text, parse_trees, semantic_frames, discourse_relations, coref_chains = process(raw_text)
        """
        return iter((
            self.corrected_text,
            self.parse_trees,
            self.semantic_frames,
            self.discourse_relations,
            self.coref_chains
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "corrected_text": self.corrected_text,
            "spelling_diffs": self.spelling_diffs,
            "parse_trees": self.parse_trees,
            "semantic_frames": self.semantic_frames,
            "discourse_relations": self.discourse_relations,
            "coref_chains": self.coref_chains,
            "pragmatic_inferences": self.pragmatic_inferences,
            "summary_counts": self.summary_counts
        }


class SmartAssistantPipeline:
    """
    End-to-end Classical NLP Reading & Writing Assistant Pipeline.
    """

    def __init__(self):
        print("Initializing NLP Pipeline components...")
        self.spell_checker = SpellChecker()
        self.dep_parser = DependencyParser()
        self.cyk_parser = CYKParser()
        self.srl_extractor = SRLExtractor()
        self.wsd = LeskWSD()
        self.coref_resolver = CoreferenceResolver()
        self.discourse_detector = DiscourseRelationDetector()
        self.pragmatic_analyzer = PragmaticInferenceAnalyzer()
        print("Pipeline initialized successfully.")

    def process(self, raw_text: str) -> PipelineResult:
        """
        Executes stages 1 through 4 in one continuous pipeline.
        
        Flow:
          Raw Paragraph -> Stage 1 (Spell Check & Diff)
                        -> Stage 2 (Dependency & POS Syntax Trees)
                        -> Stage 3 (SRL Frames, NER, WSD)
                        -> Stage 4 (Coref Chains, Discourse Relations, Pragmatic Inferences)
        """
        # --- Stage 1: Spell Checking ---
        spell_res = self.spell_checker.check(raw_text)
        corrected_text = spell_res["corrected_text"]
        spelling_diffs = spell_res["diffs"]

        # --- Stage 2: Syntactic Processing ---
        # Run syntax parsing on the spell-corrected text
        parse_trees = self.dep_parser.parse_paragraph(corrected_text)

        # --- Stage 3: Semantic Analysis ---
        semantic_frames = self.srl_extractor.process_paragraph(corrected_text)

        # --- Stage 4: Discourse & Pragmatic Processing ---
        coref_res = self.coref_resolver.resolve(corrected_text)
        discourse_res = self.discourse_detector.detect_relations(corrected_text)
        pragmatic_inferences = self.pragmatic_analyzer.analyze_paragraph(corrected_text)

        # Calculate metrics
        total_entities = sum(len(f.get("entities", [])) for f in semantic_frames)
        summary_counts = {
            "num_spelling_corrections": len(spelling_diffs),
            "num_entities": total_entities,
            "num_coref_chains": coref_res.get("num_chains", 0),
            "num_discourse_relations": discourse_res.get("num_relations", 0),
            "num_pragmatic_inferences": len(pragmatic_inferences)
        }

        return PipelineResult(
            raw_text=raw_text,
            corrected_text=corrected_text,
            spelling_diffs=spelling_diffs,
            parse_trees=parse_trees,
            semantic_frames=semantic_frames,
            discourse_relations=discourse_res.get("relations", []),
            coref_chains=coref_res.get("chains", []),
            pragmatic_inferences=pragmatic_inferences,
            summary_counts=summary_counts
        )

    def evaluate_samples(self, samples_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Runs the full pipeline on all test samples and returns a summary list.
        """
        if samples_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            samples_dir = os.path.join(base_dir, "data", "samples")

        sample_files = sorted(glob.glob(os.path.join(samples_dir, "sample_*.txt")))
        results = []

        print(f"\nEvaluating {len(sample_files)} samples in {samples_dir}...")
        for filepath in sample_files:
            filename = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            res = self.process(content)
            results.append({
                "sample": filename,
                "corrections": res.summary_counts["num_spelling_corrections"],
                "entities": res.summary_counts["num_entities"],
                "coref_chains": res.summary_counts["num_coref_chains"],
                "discourse_relations": res.summary_counts["num_discourse_relations"],
                "pragmatic_inferences": res.summary_counts["num_pragmatic_inferences"],
                "pipeline_result": res
            })

        return results

    def format_summary_table(self, eval_results: List[Dict[str, Any]]) -> str:
        """Formats evaluation metrics into a clean markdown table."""
        lines = [
            "| Sample File | # Spelling Corrections | # Entities Found | # Coref Chains | # Discourse Relations | # Pragmatic Inferences |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |"
        ]
        for r in eval_results:
            lines.append(
                f"| `{r['sample']}` | {r['corrections']} | {r['entities']} | {r['coref_chains']} | {r['discourse_relations']} | {r['pragmatic_inferences']} |"
            )
        return "\n".join(lines)
