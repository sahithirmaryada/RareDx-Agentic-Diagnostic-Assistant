import unittest

from scripts.ingest_semantic import DiseaseProfile, propagate_related_profile_signal


class SemanticIngestTests(unittest.TestCase):
    def test_propagates_subtype_signal_to_broad_parent_profile(self):
        profiles = {
            "666": DiseaseProfile(
                orphacode="666",
                disease="Osteogenesis imperfecta",
            ),
            "216812": DiseaseProfile(
                orphacode="216812",
                disease="Osteogenesis imperfecta type 3",
                hpo_terms=["Recurrent bone fractures"],
                hpo_ids=["HP:0002757"],
                genes=["COL1A1"],
                pmids=["123456"],
            ),
        }

        propagate_related_profile_signal(profiles)

        self.assertEqual(profiles["666"].genes, ["COL1A1"])
        self.assertEqual(profiles["666"].hpo_terms, ["Recurrent bone fractures"])
        self.assertEqual(profiles["666"].pmids, ["123456"])

    def test_propagates_long_single_token_parent_profile(self):
        profiles = {
            "716": DiseaseProfile(
                orphacode="716",
                disease="Phenylketonuria",
            ),
            "293284": DiseaseProfile(
                orphacode="293284",
                disease="Tetrahydrobiopterin-responsive phenylketonuria",
                hpo_terms=["Phenylalaninuria"],
                hpo_ids=["HP:0032348"],
                genes=["PAH"],
                pmids=["654321"],
            ),
        }

        propagate_related_profile_signal(profiles)

        self.assertEqual(profiles["716"].genes, ["PAH"])
        self.assertEqual(profiles["716"].hpo_terms, ["Phenylalaninuria"])


if __name__ == "__main__":
    unittest.main()
