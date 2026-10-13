"""
Simplified Lesk Algorithm for Word-Sense Disambiguation (WSD).
Disambiguates polysemous words (e.g., 'bank', 'bat', 'light', 'plant', 'star')
using WordNet gloss/example overlap with surrounding context.
"""
import re
from typing import List, Dict, Any, Optional, Set
from nltk.corpus import wordnet as wn

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours"
}


class LeskWSD:
    """
    Implements Simplified Lesk WSD using WordNet synset definitions, examples, and hypernyms.
    """

    AMBIGUOUS_WORDS = {"bank", "bat", "light", "plant", "star"}

    def _tokenize_content_words(self, text: str) -> Set[str]:
        """Extracts lowercased alphanumeric content words excluding stopwords."""
        words = re.findall(r"[A-Za-z]+", text.lower())
        return {w for w in words if w not in STOPWORDS and len(w) > 2}

    def disambiguate_word(self, word: str, context_sentence: str) -> Dict[str, Any]:
        """
        Applies Simplified Lesk to disambiguate a single word in a given sentence context.
        """
        synsets = wn.synsets(word.lower(), pos=wn.NOUN)
        if not synsets:
            synsets = wn.synsets(word.lower())
        if not synsets:
            return {
                "word": word,
                "synset": None,
                "definition": "No synset found",
                "score": 0,
                "overlap_words": []
            }

        context_words = self._tokenize_content_words(context_sentence)
        context_words.discard(word.lower())

        candidates = []
        for idx, syn in enumerate(synsets):
            # Signature = gloss definition + example sentences
            signature_text = syn.definition() + " " + " ".join(syn.examples())
            # Expanded Lesk: include hypernym definitions
            for hyper in syn.hypernyms():
                signature_text += " " + hyper.definition()

            sig_words = self._tokenize_content_words(signature_text)
            overlap = context_words.intersection(sig_words)
            score = len(overlap)

            candidates.append({
                "synset_name": syn.name(),
                "definition": syn.definition(),
                "examples": syn.examples(),
                "overlap": list(overlap),
                "score": score,
                "rank": idx
            })

        # Sort primarily by overlap score descending, secondarily by WordNet frequency rank ascending
        candidates.sort(key=lambda c: (-c["score"], c["rank"]))
        best = candidates[0]

        return {
            "word": word,
            "selected_synset": best["synset_name"],
            "definition": best["definition"],
            "score": best["score"],
            "overlap_words": best["overlap"],
            "all_candidates": candidates[:5]
        }

    def analyze_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        """
        Detects any ambiguous target words in the sentence and resolves their senses.
        """
        words = re.findall(r"[A-Za-z]+", sentence.lower())
        results = []
        for w in words:
            if w in self.AMBIGUOUS_WORDS:
                res = self.disambiguate_word(w, sentence)
                results.append(res)
        return results
