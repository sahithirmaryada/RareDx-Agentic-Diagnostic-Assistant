import unittest

from components.graph_visualizer import (
    format_relationship,
    graph_dom_id,
    parse_graph_paths,
)


class GraphVisualizerTests(unittest.TestCase):
    def test_parse_graph_paths_handles_spacing_and_dedupes_edges(self):
        graph_data = parse_graph_paths(
            [
                "Disease[Pompe Disease] -[:HAS_SYMPTOM]-> Symptom[Muscle weakness]",
                "Disease[Pompe Disease]-[:ASSOCIATED_GENE]->Gene[GAA]",
                "Disease[Pompe Disease] -[:ASSOCIATED_GENE]-> Gene[GAA]",
            ]
        )

        labels = {node["label"] for node in graph_data["nodes"]}
        groups = {node["label"]: node["group"] for node in graph_data["nodes"]}
        edge_labels = {edge["label"] for edge in graph_data["edges"]}

        self.assertEqual(labels, {"Pompe Disease", "Muscle weakness", "GAA"})
        self.assertEqual(groups["Pompe Disease"], "Disease")
        self.assertEqual(groups["Muscle weakness"], "Symptom")
        self.assertEqual(groups["GAA"], "Gene")
        self.assertEqual(len(graph_data["edges"]), 2)
        self.assertEqual(edge_labels, {"Has Symptom", "Associated Gene"})

    def test_parse_graph_paths_reports_unparseable_paths(self):
        graph_data = parse_graph_paths(["not a relationship"])

        self.assertEqual(graph_data["nodes"], [])
        self.assertEqual(graph_data["edges"], [])
        self.assertEqual(graph_data["skipped_path_count"], 1)

    def test_graph_dom_id_is_safe_and_stable(self):
        graph_data = parse_graph_paths(
            ["Disease[A] -[:HAS_SYMPTOM]-> Symptom[B]"]
        )

        self.assertEqual(graph_dom_id(graph_data, "candidate 1"), "neo4j-graph-candidate-1")
        self.assertEqual(graph_dom_id(graph_data), graph_dom_id(graph_data))

    def test_format_relationship_title_cases_neo4j_relationships(self):
        self.assertEqual(format_relationship("ASSOCIATED_GENE"), "Associated Gene")


if __name__ == "__main__":
    unittest.main()
