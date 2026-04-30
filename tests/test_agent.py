import unittest
from unittest.mock import Mock, patch

from core.agent import rare_dx_agent, validate_fusion_weights


class RareDxAgentTests(unittest.TestCase):
    @patch(
        "core.agent.resolve_icd10_code",
        return_value={
            "matched_code": "E74.02",
            "mapping_relation": "Attributed code",
        },
    )
    @patch("core.agent.resolve_citations", return_value=[{"pmid": "123456", "title": "Test Article", "abstract": "Test abstract", "year": "2023", "url": "https://pubmed.ncbi.nlm.nih.gov/123456/"}])
    @patch("core.agent.get_hybrid_retriever")
    def test_workflow_preserves_clinician_inputs_and_emits_ui_schema(
        self,
        mock_get_retriever,
        mock_resolve_citations,
        mock_resolve_icd10_code,
    ):
        fake_retriever = Mock()
        fake_retriever.query_graph.return_value = [
            {
                "disease": "Pompe Disease",
                "orphacode": "365",
                "score": 1.0,
                "graph_paths": [
                    "Disease[Pompe Disease] -[:ASSOCIATED_GENE]-> Gene[GAA]"
                ],
                "matched_symptoms": ["Muscle weakness"],
                "matched_genes": ["GAA"],
            }
        ]
        fake_retriever.query_semantic.return_value = [
            {
                "disease": "Pompe Disease",
                "orphacode": "365",
                "score": 0.8,
                "summary": "Disease: Pompe Disease. Definition: Glycogen storage disorder.",
            }
        ]
        mock_get_retriever.return_value = fake_retriever

        final_state = rare_dx_agent.invoke(
            {
                "clinical_note": "Progressive weakness with elevated CK",
                "selected_symptoms": ["Muscle weakness"],
                "selected_genes": ["gaa"],
                "graph_results": [],
                "semantic_results": [],
                "candidates": [],
                "validated_evidence": [],
                "final_report": {},
                "audit_trail": [],
            }
        )

        self.assertEqual(final_state["selected_symptoms"], ["Muscle weakness"])
        self.assertEqual(final_state["selected_genes"], ["GAA"])
        self.assertEqual(final_state["candidates"][0]["icd10"], "E74.02")
        self.assertEqual(
            final_state["candidates"][0]["graph_paths"],
            ["Disease[Pompe Disease] -[:ASSOCIATED_GENE]-> Gene[GAA]"],
        )
        self.assertEqual(final_state["candidates"][0]["citations"], [{"pmid": "123456", "title": "Test Article", "abstract": "Test abstract", "year": "2023", "url": "https://pubmed.ncbi.nlm.nih.gov/123456/"}])
        self.assertEqual(
            final_state["final_report"]["traceability_map"]["graph_paths"],
            ["Disease[Pompe Disease] -[:ASSOCIATED_GENE]-> Gene[GAA]"],
        )
        self.assertEqual(
            final_state["final_report"]["traceability_map"]["citations"],
            ["123456"],
        )
        self.assertIn(
            "ICD-10 mapped: E74.02",
            final_state["final_report"]["traceability_map"]["mcp_validation"],
        )
        fusion_events = [
            event
            for event in final_state["audit_trail"]
            if event.get("node") == "FusionGuard"
        ]
        self.assertEqual(len(fusion_events), 1)
        self.assertIn("Weights graph=", fusion_events[0]["details"])
        self.assertIn("semantic=", fusion_events[0]["details"])
        mock_resolve_citations.assert_called_with("Pompe Disease", "365")
        mock_resolve_icd10_code.assert_called_with("Pompe Disease", "365")
        fake_retriever.query_semantic.assert_called_with(
            "Progressive weakness with elevated CK",
            ["Muscle weakness"],
            ["GAA"],
        )

    @patch("core.agent.resolve_icd10_code", return_value={"matched_code": None})
    @patch("core.agent.resolve_citations", return_value=[{"pmid": "111111", "title": "Test Article", "abstract": "Test abstract", "year": "2023", "url": "https://pubmed.ncbi.nlm.nih.gov/111111/"}])
    @patch("core.agent.get_hybrid_retriever")
    def test_workflow_resolves_citations_for_graph_only_candidates(
        self,
        mock_get_retriever,
        mock_resolve_citations,
        mock_resolve_icd10_code,
    ):
        fake_retriever = Mock()
        fake_retriever.query_graph.return_value = [
            {
                "disease": "Pompe Disease",
                "orphacode": "365",
                "score": 1.0,
                "graph_paths": [
                    "Disease[Pompe Disease] -[:ASSOCIATED_GENE]-> Gene[GAA]"
                ],
                "matched_symptoms": ["Muscle weakness"],
                "matched_genes": ["GAA"],
            }
        ]
        fake_retriever.query_semantic.return_value = []
        mock_get_retriever.return_value = fake_retriever

        final_state = rare_dx_agent.invoke(
            {
                "clinical_note": "Progressive weakness with elevated CK",
                "selected_symptoms": ["Muscle weakness"],
                "selected_genes": ["gaa"],
                "graph_results": [],
                "semantic_results": [],
                "candidates": [],
                "validated_evidence": [],
                "final_report": {},
                "audit_trail": [],
            }
        )

        self.assertEqual(final_state["candidates"][0]["citations"], [{"pmid": "111111", "title": "Test Article", "abstract": "Test abstract", "year": "2023", "url": "https://pubmed.ncbi.nlm.nih.gov/111111/"}])
        self.assertEqual(
            final_state["final_report"]["traceability_map"]["citations"],
            ["111111"],
        )
        mock_resolve_citations.assert_called_with("Pompe Disease", "365")
        mock_resolve_icd10_code.assert_called_once()

    @patch("core.agent.resolve_icd10_code", return_value={"matched_code": None})
    @patch("core.agent.resolve_citations", return_value=[])
    @patch("core.agent.get_hybrid_retriever")
    def test_workflow_drops_candidates_without_graph_or_literature_support(
        self,
        mock_get_retriever,
        mock_resolve_citations,
        mock_resolve_icd10_code,
    ):
        fake_retriever = Mock()
        fake_retriever.query_graph.return_value = []
        fake_retriever.query_semantic.return_value = [
            {
                "disease": "Unverified Disease",
                "orphacode": "999999",
                "score": 0.9,
                "summary": "Disease: Unverified Disease. Definition: Placeholder.",
            }
        ]
        mock_get_retriever.return_value = fake_retriever

        final_state = rare_dx_agent.invoke(
            {
                "clinical_note": "Nonspecific note",
                "selected_symptoms": ["Fatigue"],
                "selected_genes": [],
                "graph_results": [],
                "semantic_results": [],
                "candidates": [],
                "validated_evidence": [],
                "final_report": {},
                "audit_trail": [],
            }
        )

        self.assertEqual(final_state["candidates"], [])
        self.assertEqual(
            final_state["final_report"]["summary"],
            "No grounded candidate met the RareDx evidence requirements.",
        )
        self.assertEqual(
            final_state["final_report"]["traceability_map"],
            {"graph_paths": [], "citations": [], "mcp_validation": []},
        )
        mock_resolve_citations.assert_called_with("Unverified Disease", "999999")
        mock_resolve_icd10_code.assert_not_called()

    @patch(
        "core.agent.resolve_icd10_code",
        return_value={
            "matched_code": "E74.02",
            "mapping_relation": "Attributed code",
        },
    )
    @patch("core.agent.resolve_citations", return_value=[])
    @patch("core.agent.get_hybrid_retriever")
    def test_workflow_keeps_semantic_candidate_with_orphanet_source_pmids(
        self,
        mock_get_retriever,
        mock_resolve_citations,
        mock_resolve_icd10_code,
    ):
        fake_retriever = Mock()
        fake_retriever.query_graph.return_value = []
        fake_retriever.query_semantic.return_value = [
            {
                "disease": "Pompe Disease",
                "orphacode": "365",
                "score": 0.91,
                "summary": "Disease: Pompe Disease. Phenotypes: Muscle weakness.",
                "matched_symptoms": ["Muscle weakness"],
                "matched_genes": ["GAA"],
                "source_pmids": ["20301438"],
            }
        ]
        mock_get_retriever.return_value = fake_retriever

        final_state = rare_dx_agent.invoke(
            {
                "clinical_note": "Progressive weakness with elevated CK",
                "selected_symptoms": ["Muscle weakness"],
                "selected_genes": ["gaa"],
                "graph_results": [],
                "semantic_results": [],
                "candidates": [],
                "validated_evidence": [],
                "final_report": {},
                "audit_trail": [],
                "ranking_strategy": "semantic_only",
            }
        )

        self.assertEqual(len(final_state["candidates"]), 1)
        self.assertEqual(final_state["candidates"][0]["disease"], "Pompe Disease")
        self.assertEqual(final_state["candidates"][0]["citations"][0]["pmid"], "20301438")
        self.assertEqual(final_state["candidates"][0]["matched_genes"], ["GAA"])
        self.assertEqual(
            final_state["final_report"]["traceability_map"]["citations"],
            ["20301438"],
        )
        mock_resolve_citations.assert_called_with("Pompe Disease", "365")
        mock_resolve_icd10_code.assert_called_with("Pompe Disease", "365")

    @patch(
        "core.agent.resolve_icd10_code",
        return_value={
            "matched_code": "E74.0",
            "mapping_relation": "Inclusion term",
        },
    )
    @patch("core.agent.resolve_citations", return_value=[{"pmid": "20301438", "title": "Test Article 1", "abstract": "Test abstract 1", "year": "2010", "url": "https://pubmed.ncbi.nlm.nih.gov/20301438/"}, {"pmid": "30117059", "title": "Test Article 2", "abstract": "Test abstract 2", "year": "2018", "url": "https://pubmed.ncbi.nlm.nih.gov/30117059/"}])
    @patch("core.agent.get_hybrid_retriever")
    def test_workflow_merges_graph_and_semantic_results_by_orphacode(
        self,
        mock_get_retriever,
        mock_resolve_citations,
        mock_resolve_icd10_code,
    ):
        fake_retriever = Mock()
        fake_retriever.query_graph.return_value = [
            {
                "disease": "Pompe Disease",
                "orphacode": "365",
                "score": 1.0,
                "graph_paths": [
                    "Disease[Pompe Disease] -[:ASSOCIATED_GENE]-> Gene[GAA]"
                ],
                "matched_symptoms": ["Muscle weakness"],
                "matched_genes": ["GAA"],
            }
        ]
        fake_retriever.query_semantic.return_value = [
            {
                "disease": "Glycogen storage disease due to acid maltase deficiency",
                "orphacode": "365",
                "score": 0.8,
                "summary": "Disease: Glycogen storage disease due to acid maltase deficiency. Definition: Orphanet summary.",
            }
        ]
        mock_get_retriever.return_value = fake_retriever

        final_state = rare_dx_agent.invoke(
            {
                "clinical_note": "Progressive weakness with elevated CK",
                "selected_symptoms": ["Muscle weakness"],
                "selected_genes": ["gaa"],
                "graph_results": [],
                "semantic_results": [],
                "candidates": [],
                "validated_evidence": [],
                "final_report": {},
                "audit_trail": [],
            }
        )

        self.assertEqual(len(final_state["candidates"]), 1)
        self.assertEqual(final_state["candidates"][0]["disease"], "Pompe Disease")
        self.assertEqual(final_state["candidates"][0]["citations"], [{"pmid": "20301438", "title": "Test Article 1", "abstract": "Test abstract 1", "year": "2010", "url": "https://pubmed.ncbi.nlm.nih.gov/20301438/"}, {"pmid": "30117059", "title": "Test Article 2", "abstract": "Test abstract 2", "year": "2018", "url": "https://pubmed.ncbi.nlm.nih.gov/30117059/"}])
        self.assertEqual(
            final_state["final_report"]["traceability_map"]["citations"],
            ["20301438", "30117059"],
        )
        mock_resolve_citations.assert_called_with(
            "Glycogen storage disease due to acid maltase deficiency",
            "365",
        )
        mock_resolve_icd10_code.assert_called_with("Pompe Disease", "365")

    def test_fusion_weight_validation_requires_normalized_pair(self):
        validate_fusion_weights(0.7, 0.3)

        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            validate_fusion_weights(0.7, 0.2)

        with self.assertRaisesRegex(ValueError, "between 0.0 and 1.0"):
            validate_fusion_weights(1.2, -0.2)


if __name__ == "__main__":
    unittest.main()
