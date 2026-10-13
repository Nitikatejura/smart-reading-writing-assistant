"""
N-gram Language Model for Context-Aware Disambiguation.
Computes bigram probabilities with Laplace / add-k smoothing to select
the best spelling candidate given surrounding left and right tokens.
"""
import math
from typing import Dict, List, Tuple, Optional


class NGramContextModel:
    """
    Bigram and Unigram probability estimator for context-aware spell disambiguation.
    Resolves confusing homophone pairs (e.g., 'their' vs 'there', 'your' vs 'you are').
    """

    CONFUSION_SETS = {
        "their": ["there", "they're", "their"],
        "there": ["their", "they're", "there"],
        "they're": ["their", "there", "they're"],
        "your": ["you're", "you", "your"],
        "you're": ["your", "you're"],
        "hear": ["here", "hear"],
        "here": ["hear", "here"],
        "its": ["it's", "its"],
        "it's": ["its", "it's"],
        "to": ["too", "two", "to"],
        "too": ["to", "two", "too"],
        "two": ["to", "too", "two"],
        "weather": ["whether", "weather"],
        "whether": ["weather", "whether"],
        "accept": ["except", "accept"],
        "except": ["accept", "except"],
        "witch": ["which", "witch"],
        "which": ["witch", "which"]
    }

    def __init__(self, unigram_counts: Dict[str, int], bigram_counts: Dict[Tuple[str, str], int]):
        self.unigrams = unigram_counts
        self.bigrams = bigram_counts
        self.vocab_size = max(len(unigram_counts), 1)
        self.total_tokens = sum(unigram_counts.values()) if unigram_counts else 1

    def is_ambiguous(self, word: str) -> bool:
        """Checks if a word belongs to known confusion sets."""
        return word.lower() in self.CONFUSION_SETS

    def get_confusion_candidates(self, word: str) -> List[str]:
        """Returns ambiguous homophone candidates for the word."""
        return self.CONFUSION_SETS.get(word.lower(), [word.lower()])

    def bigram_prob(self, w1: str, w2: str, alpha: float = 0.05) -> float:
        """
        Calculates Laplace smoothed bigram probability P(w2 | w1).
        P(w2 | w1) = (count(w1, w2) + alpha) / (count(w1) + alpha * V)
        """
        pair = (w1.lower(), w2.lower())
        bigram_cnt = self.bigrams.get(pair, 0)
        unigram_cnt = self.unigrams.get(w1.lower(), 0)

        prob = (bigram_cnt + alpha) / (unigram_cnt + alpha * self.vocab_size)
        return prob

    def score_candidate(self, candidate: str, prev_word: Optional[str], next_word: Optional[str]) -> float:
        """
        Calculates log probability of candidate in its local context:
        log P(cand | prev) + log P(next | cand) + log P(cand)
        """
        log_prob = 0.0
        cand_lower = candidate.lower()

        # Prior unigram log probability
        unigram_cnt = self.unigrams.get(cand_lower, 1)
        prior_prob = unigram_cnt / self.total_tokens
        log_prob += math.log(max(prior_prob, 1e-12))

        # Left context P(cand | prev)
        if prev_word:
            p_left = self.bigram_prob(prev_word, cand_lower)
            log_prob += 2.0 * math.log(max(p_left, 1e-12))

        # Right context P(next | cand)
        if next_word:
            p_right = self.bigram_prob(cand_lower, next_word)
            log_prob += 2.0 * math.log(max(p_right, 1e-12))

        return log_prob

    def pick_best_candidate(
        self,
        candidates: List[str],
        prev_word: Optional[str],
        next_word: Optional[str]
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Scores all candidates using surrounding n-gram context and returns the winner.
        """
        scores = {}
        for cand in candidates:
            score = self.score_candidate(cand, prev_word, next_word)
            scores[cand] = score

        best_cand = max(scores.keys(), key=lambda c: scores[c])
        return best_cand, scores[best_cand], scores
