import json
import unittest

from core.retriever import (
    HybridRetriever,
    build_semantic_query,
    semantic_rerank_score,
)


class FakeEmbedding:
    def tolist(self):
        return [[0.1, 0.2, 0.3]]


class FakeModel:
    def __init__(self):
        self.encoded_texts = []

    def encode(self, texts):
        self.encoded_texts = texts
        return FakeEmbedding()


class FakeCollection:
    def __init__(self):
        self.query_kwargs = None

    def query(self, **kwargs):
        self.query_kwargs = kwargs
        return {
            "documents": [[
                "Disease: Legacy Wrong Name. Phenotypes: Progressive muscle weakness."
            ]],
            "metadatas": [[
                {
                    "disease": "Pompe Disease",
                    "orphacode": "365",
                    "hpo_terms_json": json.dumps(
                        ["Progressive muscle weakness", "Elevated circulating creatine kinase concentration"]
                    ),
                    "genes_json": json.dumps(["GAA"]),
                    "pmids_json": json.dumps(["20301438"]),
                }
            ]],
            "distances": [[0.25]],
        }

    def get(self, **kwargs):
        return {"documents": [], "metadatas": []}


class HybridRetrieverSemanticTests(unittest.TestCase):
    def test_build_semantic_query_combines_structured_signal_and_note(self):
        query = build_semantic_query(
            "Progressive proximal weakness with elevated CK.",
            selected_symptoms=["Progressive muscle weakness", "Progressive muscle weakness"],
            selected_genes=["gaa"],
        )

        self.assertEqual(
            query,
            "Phenotypes: Progressive muscle weakness. "
            "Genes: GAA. "
            "Clinical note: Progressive proximal weakness with elevated CK.",
        )

    def test_query_semantic_uses_metadata_and_returns_matched_signal(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.model = FakeModel()
        retriever.collection = FakeCollection()

        results = retriever.query_semantic(
            "Progressive proximal weakness with elevated CK.",
            selected_symptoms=["Progressive muscle weakness"],
            selected_genes=["gaa"],
            limit=3,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["disease"], "Pompe Disease")
        self.assertEqual(results[0]["orphacode"], "365")
        self.assertEqual(results[0]["matched_symptoms"], ["Progressive muscle weakness"])
        self.assertEqual(results[0]["matched_genes"], ["GAA"])
        self.assertEqual(results[0]["source_pmids"], ["20301438"])
        self.assertEqual(
            retriever.model.encoded_texts,
            [
                "Phenotypes: Progressive muscle weakness. "
                "Genes: GAA. "
                "Clinical note: Progressive proximal weakness with elevated CK."
            ],
        )
        self.assertGreaterEqual(retriever.collection.query_kwargs["n_results"], 3)


class StructuredHitCollection(FakeCollection):
    def query(self, **kwargs):
        self.query_kwargs = kwargs
        return {
            "documents": [[
                "Disease: Phenotype neighbor. Phenotypes: Muscle weakness."
            ]],
            "metadatas": [[
                {
                    "disease": "Phenotype neighbor",
                    "orphacode": "999",
                    "hpo_terms_json": json.dumps(["Muscle weakness"]),
                    "genes_json": json.dumps([]),
                    "pmids_json": json.dumps(["111111"]),
                }
            ]],
            "distances": [[0.05]],
        }

    def get(self, **kwargs):
        return {
            "documents": [
                "Disease: Pompe Disease. Associated genes: GAA. Phenotypes: Muscle weakness."
            ],
            "metadatas": [
                {
                    "disease": "Pompe Disease",
                    "orphacode": "365",
                    "hpo_terms_json": json.dumps(["Muscle weakness"]),
                    "genes_json": json.dumps(["GAA"]),
                    "pmids_json": json.dumps(["20301438"]),
                }
            ],
        }


class HybridRetrieverStructuredRerankTests(unittest.TestCase):
    def test_query_semantic_promotes_exact_structured_gene_hits(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.model = FakeModel()
        retriever.collection = StructuredHitCollection()

        results = retriever.query_semantic(
            "Progressive proximal weakness.",
            selected_symptoms=["Muscle weakness"],
            selected_genes=["GAA"],
            limit=2,
        )

        self.assertEqual(results[0]["disease"], "Pompe Disease")
        self.assertEqual(results[0]["matched_genes"], ["GAA"])
        self.assertEqual(results[0]["matched_symptoms"], ["Muscle weakness"])

    def test_semantic_rerank_prioritizes_complete_phenotype_over_distractor_genes(self):
        true_score = semantic_rerank_score(
            vector_score=0.0,
            vector_rank=None,
            selected_symptoms=[
                "Waddling gait",
                "Elevated circulating creatine kinase concentration",
                "Progressive muscle weakness",
                "Calf muscle hypertrophy",
                "Cardiomyopathy",
            ],
            selected_genes=["DMD", "FKRP", "LMNA", "GAA"],
            matched_symptoms=[
                "Waddling gait",
                "Elevated circulating creatine kinase concentration",
                "Progressive muscle weakness",
                "Calf muscle hypertrophy",
                "Cardiomyopathy",
            ],
            matched_genes=["DMD"],
        )
        distractor_score = semantic_rerank_score(
            vector_score=0.0,
            vector_rank=None,
            selected_symptoms=[
                "Waddling gait",
                "Elevated circulating creatine kinase concentration",
                "Progressive muscle weakness",
                "Calf muscle hypertrophy",
                "Cardiomyopathy",
            ],
            selected_genes=["DMD", "FKRP", "LMNA", "GAA"],
            matched_symptoms=[
                "Waddling gait",
                "Elevated circulating creatine kinase concentration",
                "Progressive muscle weakness",
                "Calf muscle hypertrophy",
                "Cardiomyopathy",
            ],
            matched_genes=["DMD", "FKRP", "LMNA"],
        )

        self.assertGreater(true_score, distractor_score)


if __name__ == "__main__":
    unittest.main()
