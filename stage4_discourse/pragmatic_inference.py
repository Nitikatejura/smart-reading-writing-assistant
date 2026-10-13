"""
Pragmatic Inference Analyzer.
Detects indirect speech acts (e.g. indirect requests disguised as yes/no questions)
using pattern matching over syntactic structures.
"""
import re
from typing import List, Dict, Any, Optional
from stage2_syntax.dependency_parser import DependencyParser


class PragmaticInferenceAnalyzer:
    """
    Identifies pragmatic speech acts where surface form differs from illocutionary intent.
    Specifically detects indirect requests (e.g., 'Could you please grant me an extension?'
    -> surface: Interrogative, pragmatic intent: Directive/Action Request).
    """

    INDIRECT_REQUEST_PATTERNS = [
        (
            r"^(?:could|can|would)\s+you\s+(?:please\s+)?(.+?)\??$",
            "Directive Request",
            "Polite modal request for action rather than an inquiry into listener ability."
        ),
        (
            r"^would\s+you\s+mind\s+(.+?)\??$",
            "Directive Request",
            "Negative politeness formula functioning as a direct action request."
        ),
        (
            r"^can\s+we\s+(.+?)\??$",
            "Proposal / Collaborative Request",
            "Inclusive modal question functioning as a meeting/action proposal."
        ),
        (
            r"^(?:i\s+was\s+wondering\s+if\s+you\s+could|is\s+it\s+possible\s+to)\s+(.+?)\??$",
            "Indirect Inquiry / Request",
            "Hedged politeness strategy making an indirect action request."
        )
    ]

    def __init__(self):
        self.dep_parser = DependencyParser()

    def analyze_sentence(self, sentence: str) -> Optional[Dict[str, Any]]:
        """
        Analyzes a single sentence for indirect speech acts.
        """
        clean_sent = sentence.strip()
        lower_sent = clean_sent.lower()

        # Check if sentence is structurally a question (ends with ? or starts with modal/aux)
        for pattern, speech_act, explanation in self.INDIRECT_REQUEST_PATTERNS:
            match = re.match(pattern, lower_sent)
            if match:
                action_phrase = match.group(1).strip()
                return {
                    "sentence": clean_sent,
                    "surface_form": "Interrogative (Yes/No Question)",
                    "pragmatic_force": speech_act,
                    "requested_action": action_phrase,
                    "interpretation": f"Actually means: '[Please] {action_phrase}'.",
                    "explanation": explanation
                }

        # Check dependency parse for modal aux + subject 'you' inverted question
        parsed = self.dep_parser.parse_sentence(clean_sent)
        arcs = parsed["dependency_arcs"]
        if arcs and len(arcs) >= 3:
            first_tok = arcs[0]["text"].lower()
            second_tok = arcs[1]["text"].lower()
            if first_tok in {"could", "would", "can"} and second_tok in {"you", "we"}:
                # Extract predicate and object
                verb = next((n["text"] for n in arcs if n["pos"].startswith("VB") and n["dep"] == "ROOT"), None)
                return {
                    "sentence": clean_sent,
                    "surface_form": "Interrogative (Syntactic Question)",
                    "pragmatic_force": "Directive Request",
                    "requested_action": f"Action around predicate '{verb}'",
                    "interpretation": f"Hearer is prompted to perform: '{clean_sent}'.",
                    "explanation": "Modal auxiliary inversion used as a conventionalized polite request."
                }

        return None

    def analyze_paragraph(self, text: str) -> List[Dict[str, Any]]:
        """Scans all sentences in a paragraph for pragmatic speech acts."""
        sentences = self.dep_parser.segment_sentences(text)
        inferences = []
        for s in sentences:
            inf = self.analyze_sentence(s)
            if inf:
                inferences.append(inf)
        return inferences
