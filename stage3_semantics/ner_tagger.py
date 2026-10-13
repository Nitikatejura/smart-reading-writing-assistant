"""
Named Entity Recognition (NER) Tagger.
Extracts named entities (PERSON, ORG, GPE, DATE/TIME) using NLTK chunking and gazetteers.
"""
import re
from typing import List, Dict, Any, Tuple


class NERTagger:
    """
    Extracts entities with span positions and types.
    Combines NLTK Named Entity Chunker with pattern-based academic/temporal entity extractors.
    """

    DATE_TIME_PATTERNS = [
        (r"\b(today|tonight|yesterday|tomorrow)\b", "DATE"),
        (r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", "DATE"),
        (r"\b(morning|afternoon|evening|night)\b", "TIME"),
        (r"\b\d{1,2}\s*(?:AM|PM|am|pm)\b", "TIME"),
        (r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\b", "DATE")
    ]

    KNOWN_ENTITIES = {
        "london": "GPE",
        "alex": "PERSON",
        "sarah": "PERSON",
        "maria": "PERSON",
        "david": "PERSON",
        "elena": "PERSON",
        "kevin": "PERSON",
        "john": "PERSON",
        "watson": "PERSON",
        "miller": "PERSON",
        "library": "FACILITY"
    }

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts named entities from raw text.
        
        Returns:
            List[Dict] with keys: text, label, start, end
        """
        entities: List[Dict[str, Any]] = []
        seen_spans = set()

        # Step 1: Honorific Person detection (e.g. Dr. Watson, Prof. Miller)
        honorific_pattern = re.compile(r"\b(?:Dr\.|Prof\.|Professor|Mr\.|Mrs\.|Ms\.)\s+([A-Z][a-z]+)\b")
        for match in honorific_pattern.finditer(text):
            span = (match.start(), match.end())
            if span not in seen_spans:
                seen_spans.add(span)
                entities.append({
                    "text": match.group(0),
                    "label": "PERSON",
                    "start": match.start(),
                    "end": match.end()
                })

        # Step 2: Temporal entities (dates / times)
        for pat, label in self.DATE_TIME_PATTERNS:
            for match in re.finditer(pat, text, re.IGNORECASE):
                span = (match.start(), match.end())
                if span not in seen_spans:
                    seen_spans.add(span)
                    entities.append({
                        "text": match.group(0),
                        "label": label,
                        "start": match.start(),
                        "end": match.end()
                    })

        # Step 3: NLTK NE Chunker
        try:
            from nltk.tokenize import word_tokenize
            from nltk.tag import pos_tag
            from nltk.chunk import ne_chunk

            words = re.findall(r"[A-Za-z0-9']+|[.,!?;:]", text)
            tagged = pos_tag(words)
            chunked = ne_chunk(tagged)

            for subtree in chunked:
                if hasattr(subtree, "label"):
                    ent_text = " ".join([token for token, tag in subtree.leaves()])
                    label = subtree.label()
                    # Find span in text
                    idx = text.find(ent_text)
                    if idx != -1:
                        span = (idx, idx + len(ent_text))
                        if span not in seen_spans:
                            seen_spans.add(span)
                            entities.append({
                                "text": ent_text,
                                "label": label if label != "GPE" else "GPE",
                                "start": idx,
                                "end": idx + len(ent_text)
                            })
        except Exception:
            pass

        # Step 4: Known entity dictionary fallback
        for match in re.finditer(r"\b[A-Z][a-z]+\b", text):
            token = match.group(0)
            token_lower = token.lower()
            if token_lower in self.KNOWN_ENTITIES:
                span = (match.start(), match.end())
                if span not in seen_spans:
                    seen_spans.add(span)
                    entities.append({
                        "text": token,
                        "label": self.KNOWN_ENTITIES[token_lower],
                        "start": match.start(),
                        "end": match.end()
                    })

        # Sort entities by start index
        entities.sort(key=lambda e: e["start"])
        return entities
