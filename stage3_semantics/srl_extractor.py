"""
Semantic Role Labeler (SRL) based on Syntactic Dependency Structures.
Extracts Who (Agent), Did What (Action), to Whom/What (Patient/Theme),
and combines with Named Entities and WSD senses into structured semantic frames.
"""
from typing import List, Dict, Any, Optional
from stage2_syntax.dependency_parser import DependencyParser
from .ner_tagger import NERTagger
from .lesk_wsd import LeskWSD


class SRLExtractor:
    """
    Extracts semantic roles (Agent, Action, Patient, Modifiers) using
    syntactic dependency tree rules and integrates NER and WSD.
    """

    def __init__(self):
        self.dep_parser = DependencyParser()
        self.ner_tagger = NERTagger()
        self.wsd = LeskWSD()

    def _extract_phrase(self, head_node: Dict[str, Any], all_nodes: List[Dict[str, Any]]) -> str:
        """Collects determiners, modifiers, and head into a clean phrase."""
        head_id = head_node["id"]
        # Find modifiers attached to this head
        dependent_nodes = [
            n for n in all_nodes
            if n["head"] == head_id and n["dep"] in {"det", "amod", "compound", "poss"}
        ]
        phrase_nodes = sorted(dependent_nodes + [head_node], key=lambda x: x["id"])
        return " ".join([n["text"] for n in phrase_nodes])

    def extract_roles_from_dep(self, dep_arcs: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
        """
        Extracts semantic roles from dependency arcs.
        """
        root_node = next((n for n in dep_arcs if n["dep"] == "ROOT"), None)
        action = root_node["text"] if root_node else None

        agent = None
        patient = None
        modifiers = []

        # Find subject (Agent)
        subj_node = next((n for n in dep_arcs if n["dep"] in {"nsubj", "csubj"}), None)
        if subj_node:
            agent = self._extract_phrase(subj_node, dep_arcs)

        # Find direct object (Patient/Theme)
        obj_node = next((n for n in dep_arcs if n["dep"] in {"dobj", "obj"}), None)
        if obj_node:
            patient = self._extract_phrase(obj_node, dep_arcs)

        # Find prepositional phrases / modifiers
        prep_nodes = [n for n in dep_arcs if n["dep"] == "prep"]
        for p in prep_nodes:
            p_obj = next((n for n in dep_arcs if n["head"] == p["id"] and n["dep"] == "pobj"), None)
            if p_obj:
                phrase = f"{p['text']} {self._extract_phrase(p_obj, dep_arcs)}"
                modifiers.append(phrase)

        return {
            "agent": agent,
            "action": action,
            "patient": patient,
            "modifiers": ", ".join(modifiers) if modifiers else None
        }

    def process_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Extracts full semantic frame for a sentence:
        {
            "sentence": "...",
            "agent": "...",
            "action": "...",
            "patient": "...",
            "modifiers": "...",
            "entities": [...],
            "wsd": [...]
        }
        """
        parsed = self.dep_parser.parse_sentence(sentence)
        roles = self.extract_roles_from_dep(parsed["dependency_arcs"])
        entities = self.ner_tagger.extract_entities(sentence)
        wsd_results = self.wsd.analyze_sentence(sentence)

        return {
            "sentence": sentence,
            "agent": roles["agent"] or "None (Implied/Passive)",
            "action": roles["action"] or "None",
            "patient": roles["patient"] or "None",
            "modifiers": roles["modifiers"],
            "entities": entities,
            "wsd": wsd_results
        }

    def process_paragraph(self, text: str) -> List[Dict[str, Any]]:
        """Extracts semantic frames for all sentences in a text."""
        sentences = self.dep_parser.segment_sentences(text)
        return [self.process_sentence(s) for s in sentences]
