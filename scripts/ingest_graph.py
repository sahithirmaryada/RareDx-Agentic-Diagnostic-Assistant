import xml.etree.ElementTree as ET
import logging
from neo4j import GraphDatabase
from core.config import config

logger = logging.getLogger(__name__)

class RareDxIngestor:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            session.run(query, parameters)

    # 2. Parse Disease & Symptoms (en_product4.xml)
    def ingest_symptoms(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for disorder in root.findall(".//Disorder"):
            orphacode = disorder.find("OrphaCode").text
            name = disorder.find("Name").text

            # Create/Update Disease Node
            self.run_query("""
                MERGE (d:Disease {orphacode: $orphacode})
                SET d.name = $name
            """, {"orphacode": orphacode, "name": name})

            # Process HPO Associations
            associations = disorder.findall(".//HPODisorderAssociation")
            for assoc in associations:
                hpo_id = assoc.find(".//HPOId").text
                hpo_term = assoc.find(".//HPOTerm").text
                # Expert Tip: Store frequency on the relationship
                frequency = assoc.find(".//HPOFrequency/Name").text if assoc.find(".//HPOFrequency/Name") is not None else "Unknown"

                self.run_query("""
                    MATCH (d:Disease {orphacode: $orphacode})
                    MERGE (s:Symptom {hpo_id: $hpo_id})
                    SET s.name = $hpo_term
                    MERGE (d)-[r:HAS_SYMPTOM]->(s)
                    SET r.frequency = $frequency
                """, {
                    "orphacode": orphacode,
                    "hpo_id": hpo_id,
                    "hpo_term": hpo_term,
                    "frequency": frequency
                })

    # 3. Parse Disease & Genes (en_product6.xml)
    def ingest_genes(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for disorder in root.findall(".//Disorder"):
            orphacode = disorder.find("OrphaCode").text

            gene_associations = disorder.findall(".//DisorderGeneAssociation")
            for assoc in gene_associations:
                gene_symbol = assoc.find(".//Symbol").text
                gene_name = assoc.find(".//Name").text

                # Create Gene Node and Relationship
                self.run_query("""
                    MATCH (d:Disease {orphacode: $orphacode})
                    MERGE (g:Gene {symbol: $symbol})
                    SET g.name = $name
                    MERGE (d)-[:ASSOCIATED_GENE]->(g)
                """, {
                    "orphacode": orphacode,
                    "symbol": gene_symbol,
                    "name": gene_name
                })

if __name__ == "__main__":
    ingestor = RareDxIngestor(config.neo4j_uri, (config.neo4j_user, config.neo4j_password))
    try:
        logger.info("Starting ingestion...")
        ingestor.ingest_symptoms(config.resolve_path(config.orphanet_product4_xml))
        ingestor.ingest_genes(config.resolve_path(config.orphanet_product6_xml))
        logger.info("Ingestion complete. RareDx Knowledge Graph is grounded.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
    finally:
        ingestor.close()
