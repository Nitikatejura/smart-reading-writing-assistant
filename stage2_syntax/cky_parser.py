"""
From-scratch CYK (Cocke-Younger-Kasami) Chart Parser.
Demonstrates bottom-up chart parsing over a Chomsky Normal Form (CNF) grammar.
"""
from typing import List, Dict, Tuple, Optional, Any, Set
from .grammar import DEFAULT_CNF_GRAMMAR


class ParseNode:
    """Represents a node in a syntactic parse tree."""

    def __init__(self, symbol: str, left: Optional['ParseNode'] = None, right: Optional['ParseNode'] = None, terminal: Optional[str] = None):
        self.symbol = symbol
        self.left = left
        self.right = right
        self.terminal = terminal

    def to_bracket_string(self) -> str:
        """Returns the bracketed string notation e.g. (S (NP ...) (VP ...))."""
        if self.terminal is not None:
            return f"({self.symbol} {self.terminal})"
        left_str = self.left.to_bracket_string() if self.left else ""
        right_str = f" {self.right.to_bracket_string()}" if self.right else ""
        return f"({self.symbol} {left_str}{right_str})"

    def to_ascii_tree(self, prefix: str = "", is_last: bool = True) -> str:
        """Renders an ASCII visualization of the tree."""
        label = f"{self.symbol}: {self.terminal}" if self.terminal else self.symbol
        branch = "\\-- " if is_last else "+-- "
        lines = [f"{prefix}{branch}{label}"]

        children = []
        if self.left:
            children.append(self.left)
        if self.right:
            children.append(self.right)

        new_prefix = prefix + ("    " if is_last else "|   ")
        for i, child in enumerate(children):
            child_is_last = (i == len(children) - 1)
            lines.append(child.to_ascii_tree(new_prefix, child_is_last))

        return "\n".join(lines)


class CYKParser:
    """
    Bottom-up dynamic programming parser for context-free grammars in Chomsky Normal Form.
    """

    def __init__(self, grammar: Optional[Dict[str, Any]] = None):
        self.grammar = grammar or DEFAULT_CNF_GRAMMAR
        self.non_terminals: Dict[Tuple[str, str], List[str]] = {}
        self.terminals: Dict[str, List[str]] = {}
        self._index_grammar()

    def _index_grammar(self):
        """Indexes grammar productions for fast bottom-up lookup."""
        for lhs, rhs_list in self.grammar.items():
            for rhs in rhs_list:
                if isinstance(rhs, tuple) and len(rhs) == 2:
                    # Binary non-terminal rule: LHS -> B C
                    b, c = rhs
                    self.non_terminals.setdefault((b, c), []).append(lhs)
                elif isinstance(rhs, str):
                    # Terminal lexical rule: LHS -> word
                    word = rhs.lower()
                    self.terminals.setdefault(word, []).append(lhs)

    def parse(self, sentence: str) -> Dict[str, Any]:
        """
        Parses a sentence using the CYK algorithm.
        
        Returns:
            {
                "success": bool,
                "tokens": List[str],
                "num_parses": int,
                "bracket_tree": str,
                "ascii_tree": str,
                "chart_table": List[List[List[str]]]
            }
        """
        # Clean and tokenize words
        import re
        tokens = [w.lower() for w in re.findall(r"[A-Za-z]+", sentence)]
        n = len(tokens)
        if n == 0:
            return {"success": False, "tokens": [], "num_parses": 0, "bracket_tree": "", "ascii_tree": ""}

        # Chart: table[l][i] holds a dict mapping Symbol -> List[ParseNode]
        # l = substring length (1 to n), i = start index (0 to n-l)
        chart: List[List[Dict[str, List[ParseNode]]]] = [
            [{} for _ in range(n - l + 1)] for l in range(n + 1)
        ]

        # Step 1: Base case (length l = 1) -> apply terminal rules
        for i in range(n):
            word = tokens[i]
            matched_lhs = self.terminals.get(word, [])
            for lhs in matched_lhs:
                node = ParseNode(lhs, terminal=tokens[i])
                chart[1][i].setdefault(lhs, []).append(node)

        # Step 2: Recursive case (length l = 2 to n)
        for l in range(2, n + 1):
            for i in range(n - l + 1):
                # Partition substring into [i .. i+k] and [i+k .. i+l]
                for k in range(1, l):
                    left_nodes_dict = chart[k][i]
                    right_nodes_dict = chart[l - k][i + k]

                    for b_sym, b_nodes in left_nodes_dict.items():
                        for c_sym, c_nodes in right_nodes_dict.items():
                            candidate_parents = self.non_terminals.get((b_sym, c_sym), [])
                            for parent in candidate_parents:
                                for b_node in b_nodes:
                                    for c_node in c_nodes:
                                        p_node = ParseNode(parent, left=b_node, right=c_node)
                                        chart[l][i].setdefault(parent, []).append(p_node)

        # Step 3: Check if root symbol 'S' was generated for full string chart[n][0]
        root_parses = chart[n][0].get("S", [])
        success = len(root_parses) > 0

        bracket_tree = root_parses[0].to_bracket_string() if success else ""
        ascii_tree = root_parses[0].to_ascii_tree() if success else ""

        return {
            "success": success,
            "tokens": tokens,
            "num_parses": len(root_parses),
            "bracket_tree": bracket_tree,
            "ascii_tree": ascii_tree
        }
