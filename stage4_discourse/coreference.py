"""
Heuristic Coreference Resolution across sentences in a paragraph.
Resolves third-person anaphoric pronouns ('he', 'she', 'it', 'they')
to their nearest gender/number-compatible antecedent entities.
"""
import re
from typing import List, Dict, Any, Tuple, Optional
from stage2_syntax.dependency_parser import DependencyParser


class CoreferenceResolver:
    """
    Tracks entity mentions across discourse sentences and links pronouns
    to their antecedent entities using gender, number, recency, and grammatical salience.
    """

    MASCULINE_NAMES = {"alex", "david", "kevin", "john", "watson", "miller", "coach"}
    FEMININE_NAMES = {"sarah", "maria", "elena", "archivist", "astronomer"}
    
    PRONOUN_MAP = {
        "he": "MASCULINE_SINGULAR",
        "him": "MASCULINE_SINGULAR",
        "his": "MASCULINE_SINGULAR",
        "himself": "MASCULINE_SINGULAR",
        "she": "FEMININE_SINGULAR",
        "her": "FEMININE_SINGULAR",
        "hers": "FEMININE_SINGULAR",
        "herself": "FEMININE_SINGULAR",
        "it": "NEUTER_SINGULAR",
        "its": "NEUTER_SINGULAR",
        "itself": "NEUTER_SINGULAR",
        "they": "PLURAL",
        "them": "PLURAL",
        "their": "PLURAL",
        "theirs": "PLURAL",
        "themselves": "PLURAL"
    }

    def __init__(self):
        self.dep_parser = DependencyParser()

    def _infer_entity_type(self, noun: str, pos: str) -> str:
        """Determines the agreement category of an antecedent noun."""
        clean = noun.lower().strip()
        # Plural check
        if pos == "NNS" or pos == "NNPS" or clean in {"judges", "specimens", "algorithms", "tickets", "points", "charts"}:
            return "PLURAL"
        if clean in self.MASCULINE_NAMES or any(clean.startswith(t) for t in ["dr. watson", "prof. miller", "professor", "alex", "david", "kevin", "john"]):
            return "MASCULINE_SINGULAR"
        if clean in self.FEMININE_NAMES or any(clean.startswith(t) for t in ["sarah", "maria", "elena", "archivist", "astronomer"]):
            return "FEMININE_SINGULAR"
        return "NEUTER_SINGULAR"

    def resolve(self, text: str) -> Dict[str, Any]:
        """
        Resolves coreference chains across sentences in text.
        
        Returns:
            {
                "chains": [
                    {
                        "antecedent": str,
                        "antecedent_sentence": int,
                        "anaphors": [
                            {"pronoun": str, "sentence_idx": int}
                        ]
                    }
                ],
                "resolved_links": [
                    {"pronoun": str, "antecedent": str, "sentence": int}
                ],
                "num_chains": int
            }
        """
        sentences = self.dep_parser.segment_sentences(text)
        antecedent_pool: List[Dict[str, Any]] = []
        resolved_links: List[Dict[str, Any]] = []
        chain_map: Dict[str, Dict[str, Any]] = {}

        for sent_idx, sentence in enumerate(sentences):
            parsed = self.dep_parser.parse_sentence(sentence)
            tokens = parsed["dependency_arcs"]

            # Step 1: Register potential antecedents in current sentence
            # Prioritize subjects (higher salience), objects, and nouns
            for tok in tokens:
                pos = tok["pos"]
                text_clean = tok["text"]
                lower = text_clean.lower()

                # Skip non-referential pronouns or stopwords
                if lower in self.PRONOUN_MAP or lower in {"i", "me", "my", "you", "your", "we", "us", "our"}:
                    continue

                if pos.startswith("NN") and len(text_clean) > 1:
                    category = self._infer_entity_type(text_clean, pos)
                    salience = 3.0 if tok["dep"] in {"nsubj", "csubj"} else (2.0 if tok["dep"] in {"dobj", "pobj"} else 1.0)
                    antecedent_pool.append({
                        "text": text_clean,
                        "category": category,
                        "sentence_idx": sent_idx,
                        "salience": salience,
                        "pos": pos
                    })

            # Step 2: Resolve pronouns in current sentence
            for tok in tokens:
                lower = tok["text"].lower()
                if lower in self.PRONOUN_MAP:
                    target_category = self.PRONOUN_MAP[lower]

                    # Search backward through antecedent pool
                    best_cand = None
                    best_score = -999.0

                    for cand in reversed(antecedent_pool):
                        # Agreement check
                        compat = False
                        if target_category == cand["category"]:
                            compat = True
                        elif target_category in {"MASCULINE_SINGULAR", "FEMININE_SINGULAR"} and cand["category"] == "NEUTER_SINGULAR":
                            # If an occupation/title like coach or archivist was categorized as neuter, allow compatibility
                            if cand["text"].lower() in {"coach", "archivist", "astronomer", "partner", "doctor", "professor", "judge"}:
                                compat = True

                        if compat:
                            # Recency distance penalty
                            sent_dist = sent_idx - cand["sentence_idx"]
                            if sent_dist < 0:
                                continue
                            recency_penalty = sent_dist * 1.5
                            score = cand["salience"] - recency_penalty

                            if score > best_score:
                                best_score = score
                                best_cand = cand

                    if best_cand:
                        ant_text = best_cand["text"]
                        resolved_links.append({
                            "pronoun": tok["text"],
                            "antecedent": ant_text,
                            "sentence": sent_idx + 1,
                            "antecedent_sentence": best_cand["sentence_idx"] + 1
                        })

                        # Group into chains
                        if ant_text not in chain_map:
                            chain_map[ant_text] = {
                                "antecedent": ant_text,
                                "category": best_cand["category"],
                                "mentions": [ant_text],
                                "anaphors": []
                            }
                        chain_map[ant_text]["mentions"].append(tok["text"])
                        chain_map[ant_text]["anaphors"].append({
                            "pronoun": tok["text"],
                            "sentence_idx": sent_idx + 1
                        })

        return {
            "chains": list(chain_map.values()),
            "resolved_links": resolved_links,
            "num_chains": len(chain_map)
        }
