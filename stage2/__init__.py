"""
Stage 2: Syntactic Processing (Alias for stage2_syntax)
"""
from stage2_syntax.dependency_parser import DependencyParser
from stage2_syntax.cky_parser import CYKParser
from stage2_syntax.grammar import DEFAULT_CNF_GRAMMAR

__all__ = ["DependencyParser", "CYKParser", "DEFAULT_CNF_GRAMMAR"]
