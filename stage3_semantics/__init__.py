"""
Stage 3: Semantic Analysis Module
Includes Named Entity Recognition (NER), Dependency-based Semantic Role Labeling (SRL),
and Simplified Lesk Word-Sense Disambiguation (WSD).
"""
from .ner_tagger import NERTagger
from .srl_extractor import SRLExtractor
from .lesk_wsd import LeskWSD

__all__ = ["NERTagger", "SRLExtractor", "LeskWSD"]
