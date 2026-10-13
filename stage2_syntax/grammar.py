"""
Chomsky Normal Form (CNF) Grammar Definitions for CYK Parser.
Rules are formatted as either:
  A -> B C  (binary non-terminal production)
  A -> 'word' (unary terminal production)
"""
from typing import Dict, List, Tuple, Union

# Hand-crafted 10-15 CFG rules covering simple declarative sentences
DEFAULT_CNF_GRAMMAR: Dict[str, Union[List[Tuple[str, str]], List[str]]] = {
    # Binary non-terminal productions (A -> B C)
    "S": [("NP", "VP")],
    "VP": [
        ("V", "NP"),
        ("VP", "PP"),
        ("V", "PP"),
        ("V", "S")
    ],
    "NP": [
        ("Det", "N"),
        ("NP", "PP"),
        ("Adj", "N"),
        ("Det", "Nom"),
        ("N", "N")
    ],
    "Nom": [
        ("Adj", "N"),
        ("Nom", "PP")
    ],
    "PP": [
        ("P", "NP")
    ],

    # Lexical terminal productions (A -> word)
    "Det": ["the", "a", "an", "my", "her", "his", "their", "our", "this"],
    "N": [
        "astronomer", "telescope", "star", "coach", "bat", "light", "maria", "plant",
        "student", "essay", "code", "algorithm", "partner", "kevin", "alex", "david",
        "elena", "john", "watson", "miller", "bug", "specimens", "experiment", "draft",
        "stadium", "galaxy", "images", "prototype", "exhibition", "tickets", "library"
    ],
    "V": [
        "pointed", "saw", "tested", "reviewed", "wrote", "liked", "visited", "noticed",
        "captured", "logged", "refactored", "showed", "explained", "found", "managed",
        "gave", "started", "fixed"
    ],
    "P": [
        "with", "in", "near", "at", "toward", "on", "from", "for", "by", "before", "into"
    ],
    "Adj": [
        "bright", "new", "green", "small", "clear", "ancient", "initial", "distant",
        "northern", "unusual", "urgent", "major", "good"
    ]
}
