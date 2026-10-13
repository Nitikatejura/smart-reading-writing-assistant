"""
Stage 3: Semantic Analysis (Alias for stage3_semantics)
"""
from stage3_semantics.srl_extractor import SRLExtractor
from stage3_semantics.ner_tagger import NERTagger
from stage3_semantics.lesk_wsd import LeskWSD

__all__ = ["SRLExtractor", "NERTagger", "LeskWSD"]
