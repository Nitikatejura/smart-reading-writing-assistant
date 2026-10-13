"""
Stage 4: Discourse & Pragmatic Processing (Alias for stage4_discourse)
"""
from stage4_discourse.coreference import CoreferenceResolver
from stage4_discourse.discourse_relations import DiscourseRelationDetector
from stage4_discourse.pragmatic_inference import PragmaticInferenceAnalyzer

__all__ = ["CoreferenceResolver", "DiscourseRelationDetector", "PragmaticInferenceAnalyzer"]
