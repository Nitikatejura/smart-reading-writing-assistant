"""
From-scratch Levenshtein Edit Distance and Candidate Generator.
Implements dynamic programming matrix calculation without third-party libraries.
"""
from typing import List, Tuple, Dict, Set


def levenshtein_matrix(s1: str, s2: str) -> Tuple[int, List[List[int]]]:
    """
    Computes the standard Levenshtein distance dynamic programming matrix between two strings.
    
    dp[i][j] represents the minimum edit distance between s1[0..i-1] and s2[0..j-1].
    Cost operations:
      - Insertion: 1
      - Deletion: 1
      - Substitution: 0 if s1[i-1] == s2[j-1] else 1
    
    Returns:
        (distance, dp_matrix)
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    return dp[m][n], dp


def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes Damerau-Levenshtein distance (allows transpositions of adjacent characters).
    Useful for common typing errors like 'tehnique' -> 'technique' or 'recieved' -> 'received'.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
            # Transposition check
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)

    return dp[m][n]


class CandidateGenerator:
    """
    Generates potential spelling correction candidates at edit distance 1 and 2,
    then filters and ranks them against a word frequency dictionary.
    """

    LETTERS = "abcdefghijklmnopqrstuvwxyz"

    def __init__(self, vocab_frequencies: Dict[str, int]):
        self.vocab = vocab_frequencies
        self.total_count = sum(vocab_frequencies.values()) if vocab_frequencies else 1

    def edits1(self, word: str) -> Set[str]:
        """All string variations at edit distance 1 (splits, deletes, transposes, replaces, inserts)."""
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in self.LETTERS]
        inserts = [L + c + R for L, R in splits for c in self.LETTERS]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word: str) -> Set[str]:
        """All variations at edit distance 2."""
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def known(self, words: Set[str]) -> Set[str]:
        """Returns the subset of words that exist in the dictionary."""
        return set(w for w in words if w in self.vocab)

    def get_candidates(self, word: str, max_candidates: int = 10) -> List[Tuple[str, int, int]]:
        """
        Returns ranked candidates for a given word.
        Each candidate is (candidate_word, edit_distance, frequency).
        Uses Damerau-Levenshtein distance to properly treat character transpositions
        (e.g., 'recieved' -> 'received') as edit distance 1.
        """
        lower_word = word.lower()
        candidates = []

        # Check common informal/phonetic mappings first
        phonetic_slang = {
            "tonite": "tonight",
            "alot": "a lot",
            "im": "i'm",
            "u": "you",
            "ur": "your",
            "cuz": "because",
            "thru": "through"
        }
        if lower_word in phonetic_slang:
            target = phonetic_slang[lower_word]
            dist = damerau_levenshtein_distance(lower_word, target)
            return [(target, dist, self.vocab.get(target, 10000))]

        # Distance 1 candidates (including transpositions)
        e1_candidates = self.known(self.edits1(lower_word))
        for cand in e1_candidates:
            dist = damerau_levenshtein_distance(lower_word, cand)
            candidates.append((cand, dist, self.vocab.get(cand, 1)))

        # Only evaluate distance 2 if no distance 1 candidate was found in dictionary
        if not candidates:
            e2_candidates = self.known(self.edits2(lower_word))
            for cand in e2_candidates:
                dist = damerau_levenshtein_distance(lower_word, cand)
                candidates.append((cand, dist, self.vocab.get(cand, 1)))

        # Deduplicate and sort by edit distance ascending, frequency descending
        seen = set()
        deduped = []
        for cand, dist, freq in sorted(candidates, key=lambda x: (x[1], -x[2])):
            if cand not in seen:
                seen.add(cand)
                deduped.append((cand, dist, freq))
                if len(deduped) >= max_candidates:
                    break

        return deduped
