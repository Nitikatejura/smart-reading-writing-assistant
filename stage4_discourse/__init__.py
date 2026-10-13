"""
Stage 4: Discourse and Pragmatic Processing Module
Includes Coreference Resolution, Discourse Connective Relation Typing,
and Pragmatic Speech Act (Indirect Request) Inference.
"""
from .coreference import CoreferenceResolver
from .discourse_relations import DiscourseRelationDetector
from .pragmatic_inference import PragmaticInferenceAnalyzer

__all__ = ["CoreferenceResolver", "DiscourseRelationDetector", "PragmaticInferenceAnalyzer"]
