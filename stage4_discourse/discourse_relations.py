"""
Discourse Connective Detection and Relation Typing.
Classifies inter-clausal and inter-sentential discourse relations:
Contrast, Cause/Result, Elaboration/Addition, and Temporal.
"""
import re
from typing import List, Dict, Any, Tuple
from stage2_syntax.dependency_parser import DependencyParser


class DiscourseRelationDetector:
    """
    Identifies discourse markers/connectives and labels the rhetorical/discourse
    relation they signal between propositions.
    """

    # Multi-word connectives first, then single-word
    CONNECTIVES = [
        # Contrast
        ("on the other hand", "Contrast"),
        ("nevertheless", "Contrast"),
        ("however", "Contrast"),
        ("although", "Contrast"),
        ("though", "Contrast"),
        ("whereas", "Contrast"),
        ("but", "Contrast"),
        ("yet", "Contrast"),
        
        # Cause / Result
        ("as a result", "Cause/Result"),
        ("consequently", "Cause/Result"),
        ("therefore", "Cause/Result"),
        ("because", "Cause/Result"),
        ("since", "Cause/Result"),
        ("so", "Cause/Result"),
        ("thus", "Cause/Result"),
        
        # Elaboration / Addition
        ("in addition", "Elaboration"),
        ("furthermore", "Elaboration"),
        ("moreover", "Elaboration"),
        ("besides", "Elaboration"),
        ("also", "Elaboration"),
        
        # Temporal
        ("and then", "Temporal"),
        ("meanwhile", "Temporal"),
        ("subsequently", "Temporal"),
        ("then", "Temporal"),
        ("before", "Temporal"),
        ("after", "Temporal")
    ]

    def __init__(self):
        self.dep_parser = DependencyParser()

    def detect_relations(self, text: str) -> Dict[str, Any]:
        """
        Detects discourse connectives in paragraph and assigns relation types.
        
        Returns:
            {
                "relations": [
                    {
                        "connective": str,
                        "relation_type": str,
                        "sentence_idx": int,
                        "sentence": str,
                        "arg1_summary": str,
                        "arg2_summary": str
                    }
                ],
                "num_relations": int
            }
        """
        sentences = self.dep_parser.segment_sentences(text)
        relations: List[Dict[str, Any]] = []

        for sent_idx, sentence in enumerate(sentences):
            sent_lower = sentence.lower()
            matched_spans = set()

            for conn, rel_type in self.CONNECTIVES:
                # Use regex with word boundary check
                pattern = r"\b" + re.escape(conn) + r"\b"
                for match in re.finditer(pattern, sent_lower):
                    start, end = match.span()
                    # Filter out non-connective phrases like 'so far'
                    if conn == "so" and sent_lower[end:].strip().startswith("far"):
                        continue
                    if conn == "so" and sent_lower[end:].strip().startswith("much"):
                        continue

                    # Prevent overlapping sub-phrases (e.g. 'then' inside 'and then')
                    if any(s <= start and end <= e for (s, e) in matched_spans):
                        continue
                    matched_spans.add((start, end))

                    # Arg1 is prior sentence if connective starts sentence, else preceding clause
                    if start <= 3 and sent_idx > 0:
                        arg1_text = sentences[sent_idx - 1]
                        arg2_text = sentence[end:].strip(" ,;:")
                    else:
                        arg1_text = sentence[:start].strip(" ,;:")
                        arg2_text = sentence[end:].strip(" ,;:")

                    relations.append({
                        "connective": match.group(0),
                        "relation_type": rel_type,
                        "sentence_idx": sent_idx + 1,
                        "sentence": sentence,
                        "arg1": arg1_text,
                        "arg2": arg2_text
                    })

        return {
            "relations": relations,
            "num_relations": len(relations)
        }
