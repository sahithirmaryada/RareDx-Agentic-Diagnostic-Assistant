import unittest
from unittest.mock import Mock, patch

from core.agent import rare_dx_agent


class RareDxAgentTests(unittest.TestCase):
    @patch(
        "core.agent.resolve_icd10_code",
        return_value={
            "matched_code": "E74.02",
            "mapping_relation": "Attributed code",
        },
    )
    @patch("core.agent.resolve_citations", return_value=["123456"])
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
        self.assertEqual(final_state["candidates"][0]["citations"], ["123456"])
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
        mock_resolve_citations.assert_called_with("Pompe Disease", "365")
        mock_resolve_icd10_code.assert_called_with("Pompe Disease", "365")

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
            "matched_code": "E74.0",
            "mapping_relation": "Inclusion term",
        },
    )
    @patch("core.agent.resolve_citations", return_value=["20301438", "30117059"])
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
        self.assertEqual(final_state["candidates"][0]["citations"], ["20301438", "30117059"])
        self.assertEqual(
            final_state["final_report"]["traceability_map"]["citations"],
            ["20301438", "30117059"],
        )
        mock_resolve_citations.assert_called_with(
            "Glycogen storage disease due to acid maltase deficiency",
            "365",
        )
        mock_resolve_icd10_code.assert_called_with("Pompe Disease", "365")


if __name__ == "__main__":
    unittest.main()
