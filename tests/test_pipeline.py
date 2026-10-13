"""
Automated Unit Tests for the Smart Reading & Writing Assistant (Stages 1 through 5).
"""
import pytest

from stage1_spellcheck.levenshtein import levenshtein_matrix, damerau_levenshtein_distance, CandidateGenerator
from stage1_spellcheck.spell_checker import SpellChecker
from stage2_syntax.cky_parser import CYKParser
from stage2_syntax.dependency_parser import DependencyParser
from stage3_semantics.lesk_wsd import LeskWSD
from stage3_semantics.srl_extractor import SRLExtractor
from stage4_discourse.coreference import CoreferenceResolver
from stage4_discourse.discourse_relations import DiscourseRelationDetector
from stage4_discourse.pragmatic_inference import PragmaticInferenceAnalyzer
from stage5_pipeline.pipeline_runner import SmartAssistantPipeline


def test_stage1_levenshtein():
    dist, matrix = levenshtein_matrix("kitten", "sitting")
    assert dist == 3
    assert len(matrix) == 7
    assert len(matrix[0]) == 8

    # Transposition with Damerau-Levenshtein
    assert damerau_levenshtein_distance("recieved", "received") == 1
    assert damerau_levenshtein_distance("tehnique", "technique") == 1


def test_stage1_context_spellcheck():
    sc = SpellChecker()
    res = sc.check("Their is a problem with their code.")
    # 'Their is' should be corrected to 'There is', but 'their code' should remain 'their code'
    assert res["corrected_text"].startswith("There is")
    assert "their code" in res["corrected_text"]
    assert len(res["diffs"]) >= 1


def test_stage2_cky_parser():
    parser = CYKParser()
    res = parser.parse("The astronomer pointed the telescope")
    assert res["success"] is True
    assert res["num_parses"] >= 1
    assert "NP" in res["bracket_tree"]
    assert "VP" in res["bracket_tree"]


def test_stage2_dependency_parser():
    dep = DependencyParser()
    res = dep.parse_sentence("Maria visited the research plant.")
    assert len(res["tokens"]) > 0
    assert any(arc["dep"] == "ROOT" for arc in res["dependency_arcs"])
    assert any(arc["dep"] == "nsubj" for arc in res["dependency_arcs"])


def test_stage3_lesk_wsd():
    wsd = LeskWSD()
    # Test bank disambiguation
    res_river = wsd.disambiguate_word("bank", "The fish swam along the river bank.")
    assert "river" in res_river["overlap_words"] or res_river["selected_synset"] == "bank.n.01"

    res_finance = wsd.disambiguate_word("bank", "The financial bank approved the monetary loan.")
    assert res_finance["selected_synset"] == "depository_financial_institution.n.01" or res_finance["score"] > 0


def test_stage3_srl_extraction():
    srl = SRLExtractor()
    res = srl.process_sentence("David tested every component carefully.")
    assert res["action"] == "tested"
    assert "David" in res["agent"]


def test_stage4_coreference():
    resolver = CoreferenceResolver()
    text = "Kevin wrote the initial draft, and then he revised the charts."
    res = resolver.resolve(text)
    assert res["num_chains"] >= 1
    assert any(c["antecedent"] == "Kevin" for c in res["chains"])


def test_stage4_discourse_and_pragmatics():
    dr = DiscourseRelationDetector()
    text = "He worked hard; however, the bug persisted."
    rel_res = dr.detect_relations(text)
    assert any(r["relation_type"] == "Contrast" for r in rel_res["relations"])

    pi = PragmaticInferenceAnalyzer()
    prag_res = pi.analyze_sentence("Could you please review my essay?")
    assert prag_res is not None
    assert prag_res["pragmatic_force"] == "Directive Request"


def test_stage5_pipeline_integration():
    pipeline = SmartAssistantPipeline()
    sample_text = "Their is a bug. Alex said he would fix it. Could you check the draft?"
    res = pipeline.process(sample_text)
    assert res.corrected_text.startswith("There is")
    assert len(res.parse_trees) >= 2
    assert len(res.pragmatic_inferences) >= 1
    assert res.summary_counts["num_pragmatic_inferences"] >= 1


def test_stage5_process_unpacking():
    from pipeline import process
    sample_text = "Their is a bug. Alex said he would fix it. Could you check the draft?"
    corrected_text, parse_trees, semantic_frames, discourse_relations, coref_chains = process(sample_text)
    assert corrected_text.startswith("There is")
    assert isinstance(parse_trees, list)
    assert isinstance(semantic_frames, list)
    assert isinstance(discourse_relations, list)
    assert isinstance(coref_chains, list)
