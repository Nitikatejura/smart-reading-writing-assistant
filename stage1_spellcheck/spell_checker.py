"""
Stage 1: Complete Spell Checker with Context Disambiguation and Diff Generation.
"""
import os
import re
import json
from typing import List, Dict, Any, Tuple, Optional
from .levenshtein import levenshtein_matrix, CandidateGenerator
from .ngram_context import NGramContextModel


class SpellChecker:
    """
    Combines:
      1. From-scratch Levenshtein minimum edit distance candidate generator.
      2. N-gram context language model for ambiguous/confusing words.
      3. Granular diff tracking explaining exactly why each token was changed.
    """

    def __init__(self, dict_path: Optional[str] = None, bigrams_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if dict_path is None:
            dict_path = os.path.join(base_dir, "data", "dictionary.txt")
        if bigrams_path is None:
            bigrams_path = os.path.join(base_dir, "data", "bigrams.json")

        self.vocab = self._load_dictionary(dict_path)
        self.bigrams = self._load_bigrams(bigrams_path)
        self.candidate_gen = CandidateGenerator(self.vocab)
        self.ngram_model = NGramContextModel(self.vocab, self.bigrams)

    def _load_dictionary(self, path: str) -> Dict[str, int]:
        vocab = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        vocab[parts[0].lower()] = int(parts[1])
                    elif len(parts) == 1 and parts[0]:
                        vocab[parts[0].lower()] = 100
        return vocab

    def _load_bigrams(self, path: str) -> Dict[Tuple[str, str], int]:
        bigrams = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for key, count in data.items():
                    w1, w2 = key.split(" ", 1)
                    bigrams[(w1.lower(), w2.lower())] = count
        return bigrams

    def tokenize_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """Extracts words along with their character start/end offsets."""
        tokens = []
        for match in re.finditer(r"[A-Za-z0-9']+", text):
            tokens.append((match.group(), match.start(), match.end()))
        return tokens

    def _match_casing(self, original: str, corrected: str) -> str:
        """Preserves title case or upper case from the original token."""
        if original.isupper():
            return corrected.upper()
        if original and original[0].isupper():
            return corrected.capitalize()
        return corrected.lower()

    def check(self, text: str) -> Dict[str, Any]:
        """
        Runs the Stage 1 spell checker on raw text.
        
        Returns:
            {
                "original_text": str,
                "corrected_text": str,
                "diffs": [
                    {
                        "original": str,
                        "corrected": str,
                        "start": int,
                        "end": int,
                        "reason": str,
                        "edit_distance": int,
                        "scores": dict (optional)
                    }
                ],
                "num_corrections": int
            }
        """
        tokens = self.tokenize_with_spans(text)
        if not tokens:
            return {
                "original_text": text,
                "corrected_text": text,
                "diffs": [],
                "num_corrections": 0
            }

        diffs = []
        # Reconstruct string with replacements
        corrected_chars = list(text)
        # Process tokens in reverse order so character offsets remain valid during replacements
        for idx in range(len(tokens) - 1, -1, -1):
            raw_token, start, end = tokens[idx]
            lower_token = raw_token.lower()

            # Identify context neighbors (ignoring non-alphanumeric)
            prev_token = tokens[idx - 1][0].lower() if idx > 0 else None
            next_token = tokens[idx + 1][0].lower() if idx + 1 < len(tokens) else None

            # Case A: Ambiguous homophone (e.g., 'their' vs 'there')
            if self.ngram_model.is_ambiguous(lower_token):
                confusion_cands = self.ngram_model.get_confusion_candidates(lower_token)
                best_cand, best_score, score_dict = self.ngram_model.pick_best_candidate(
                    confusion_cands, prev_token, next_token
                )
                if best_cand != lower_token:
                    dist, _ = levenshtein_matrix(lower_token, best_cand)
                    replacement = self._match_casing(raw_token, best_cand)
                    diff_entry = {
                        "original": raw_token,
                        "corrected": replacement,
                        "start": start,
                        "end": end,
                        "reason": f"n-gram context score: {best_cand}={best_score:.2f} vs {lower_token}={score_dict.get(lower_token, -99.0):.2f}",
                        "edit_distance": dist,
                        "scores": {k: round(v, 2) for k, v in score_dict.items()}
                    }
                    diffs.append(diff_entry)
                    corrected_chars[start:end] = list(replacement)
                continue

            # Case B: Unknown / misspelled word or informal phonetic spelling
            informal_slang = {"tonite", "cuz", "thru", "alot", "u", "ur"}
            if lower_token.isdigit():
                continue

            # Check if valid word in dictionary or valid morphological inflection in WordNet
            is_valid_word = (lower_token in self.vocab)
            if not is_valid_word:
                try:
                    from nltk.corpus import wordnet
                    if bool(wordnet.synsets(lower_token)):
                        is_valid_word = True
                except Exception:
                    pass

            if is_valid_word and lower_token not in informal_slang:
                continue

            # Generate candidate corrections using scratch Levenshtein
            candidates = self.candidate_gen.get_candidates(lower_token, max_candidates=5)
            if not candidates:
                continue

            # Filter candidates by minimal edit distance
            min_dist = candidates[0][1]
            top_candidates = [c[0] for c in candidates if c[1] == min_dist]

            if len(top_candidates) == 1:
                chosen_cand = top_candidates[0]
                reason = f"Levenshtein edit-distance = {min_dist} (DP matrix), vocab freq = {self.vocab.get(chosen_cand, 0)}"
            else:
                # Use n-gram context to break ties among multiple edit-distance candidates
                chosen_cand, score, _ = self.ngram_model.pick_best_candidate(top_candidates, prev_token, next_token)
                reason = f"Levenshtein edit-distance = {min_dist} (tie-broken by n-gram log-prob = {score:.2f})"

            replacement = self._match_casing(raw_token, chosen_cand)
            diff_entry = {
                "original": raw_token,
                "corrected": replacement,
                "start": start,
                "end": end,
                "reason": reason,
                "edit_distance": min_dist
            }
            diffs.append(diff_entry)
            corrected_chars[start:end] = list(replacement)

        # Reverse diffs so they are in reading order
        diffs.reverse()
        corrected_text = "".join(corrected_chars)

        return {
            "original_text": text,
            "corrected_text": corrected_text,
            "diffs": diffs,
            "num_corrections": len(diffs)
        }
