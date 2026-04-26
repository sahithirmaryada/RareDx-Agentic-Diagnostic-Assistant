import os
from pathlib import Path
from functools import lru_cache
import logging
from core.config import config

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import Neo4jError, ServiceUnavailable
except ModuleNotFoundError:
    GraphDatabase = None
    Neo4jError = Exception
    ServiceUnavailable = Exception

try:
    from chromadb.errors import ChromaException
except ImportError:
    ChromaException = Exception

try:
    import chromadb
except ModuleNotFoundError:
    chromadb = None


class InfrastructureError(RuntimeError):
    pass

class HybridRetriever:
    def __init__(self):
        if GraphDatabase is None:
            raise InfrastructureError(
                "The neo4j client is not installed. Grounded diagnostic services cannot run."
            )
        if chromadb is None:
            raise InfrastructureError(
                "The chromadb client is not installed. Grounded diagnostic services cannot run."
            )

        try:
            self.driver = GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password))
            self.driver.verify_connectivity()
        except (ServiceUnavailable, Neo4jError) as exc:
            raise InfrastructureError(
                "Neo4j is unavailable. Grounded diagnostic services cannot run."
            ) from exc

        try:
            self.chroma_client = chromadb.HttpClient(host=config.chroma_host, port=config.chroma_port)
            self.collection = self.chroma_client.get_collection("rare_disease_summaries")
            self.collection.count()
        except Exception as exc:
            self.driver.close()
            raise InfrastructureError(
                "ChromaDB is unavailable. Grounded diagnostic services cannot run."
            ) from exc

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(config.biolord_model_name)
        except ModuleNotFoundError as exc:
            self._cleanup()
            raise InfrastructureError(
                "The sentence-transformers package is not installed. Grounded diagnostic services cannot run."
            ) from exc
        except Exception as exc:
            self._cleanup()
            raise InfrastructureError(
                "The semantic embedding model is unavailable. Grounded diagnostic services cannot run."
            ) from exc

    def _cleanup(self):
        try:
            if hasattr(self, "driver") and self.driver is not None:
                self.driver.close()
        except Exception:
            pass
        try:
            if hasattr(self, "chroma_client") and self.chroma_client is not None:
                close_fn = getattr(self.chroma_client, "close", None)
                if callable(close_fn):
                    close_fn()
        except Exception:
            pass

    def close(self):
        self._cleanup()

    def query_graph(self, selected_symptoms, selected_genes, limit=10):
        query = """
        MATCH (d:Disease)
        OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
        WITH d, collect(DISTINCT s.name) AS disease_symptoms
        OPTIONAL MATCH (d)-[:ASSOCIATED_GENE]->(g:Gene)
        WITH d, disease_symptoms, collect(DISTINCT g.symbol) AS disease_genes
        WITH d,
             [name IN disease_symptoms WHERE name IN $selected_symptoms] AS matched_symptoms,
             [symbol IN disease_genes WHERE symbol IN $selected_genes] AS matched_genes
        WITH d,
             matched_symptoms,
             matched_genes,
             size(matched_symptoms) AS symptom_hits,
             size(matched_genes) AS gene_hits
        WHERE symptom_hits > 0 OR gene_hits > 0
        RETURN d.name AS disease,
               d.orphacode AS orphacode,
               matched_symptoms,
               matched_genes,
               symptom_hits,
               gene_hits
        ORDER BY gene_hits DESC, symptom_hits DESC, disease
        LIMIT $limit
        """

        denominator = max(len(selected_symptoms) + len(selected_genes), 1)

        try:
            with self.driver.session() as session:
                result = session.run(
                    query,
                    {
                        "selected_symptoms": selected_symptoms,
                        "selected_genes": selected_genes,
                        "limit": limit,
                    },
                )
                rows = list(result)
        except (ServiceUnavailable, Neo4jError) as exc:
            raise InfrastructureError(
                "Neo4j query failed. Grounded diagnostic services cannot run."
            ) from exc

        graph_results = []
        for row in rows:
            matched_symptoms = [item for item in row["matched_symptoms"] if item]
            matched_genes = [item for item in row["matched_genes"] if item]
            graph_paths = [
                f'Disease[{row["disease"]}] -[:HAS_SYMPTOM]-> Symptom[{symptom}]'
                for symptom in matched_symptoms
            ]
            graph_paths.extend(
                f'Disease[{row["disease"]}] -[:ASSOCIATED_GENE]-> Gene[{gene}]'
                for gene in matched_genes
            )
            graph_results.append(
                {
                    "disease": row["disease"],
                    "orphacode": row["orphacode"],
                    "score": (row["symptom_hits"] + row["gene_hits"]) / denominator,
                    "graph_paths": graph_paths,
                    "matched_symptoms": matched_symptoms,
                    "matched_genes": matched_genes,
                }
            )

        return graph_results

    def query_semantic(self, clinical_note, limit=5):
        note = clinical_note.strip()
        if not note:
            return []

        query_embedding = self.model.encode([note])

        try:
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
        except ChromaException as exc:
            raise InfrastructureError(
                "ChromaDB query failed. Grounded diagnostic services cannot run."
            ) from exc

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        semantic_results = []
        seen = set()
        for document, metadata, distance in zip(documents, metadatas, distances):
            disease_name = _extract_disease_name(document)
            if not disease_name or disease_name in seen:
                continue
            seen.add(disease_name)
            semantic_results.append(
                {
                    "disease": disease_name,
                    "orphacode": (metadata or {}).get("orphacode"),
                    "score": 1.0 / (1.0 + max(distance, 0.0)),
                    "summary": document,
                }
            )

        return semantic_results


def _extract_disease_name(document):
    prefix = "Disease:"
    divider = ". Definition:"
    if not document or not document.startswith(prefix):
        return None
    if divider in document:
        return document[len(prefix):document.index(divider)].strip()
    return document[len(prefix):].strip()


def get_infrastructure_status():
    status = {
        "neo4j": {"ok": False, "detail": "Offline"},
        "chroma": {"ok": False, "detail": "Offline"},
        "ready": False,
    }

    driver = None
    if GraphDatabase is None:
        status["neo4j"] = {"ok": False, "detail": "neo4j client not installed"}
    else:
        try:
            driver = GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password))
            driver.verify_connectivity()
            status["neo4j"] = {"ok": True, "detail": "Online"}
        except (ServiceUnavailable, Neo4jError) as exc:
            status["neo4j"] = {"ok": False, "detail": str(exc)}
        finally:
            if driver is not None:
                driver.close()

    if chromadb is None:
        status["chroma"] = {"ok": False, "detail": "chromadb client not installed"}
    else:
        try:
            chroma_client = chromadb.HttpClient(host=config.chroma_host, port=config.chroma_port)
            collection = chroma_client.get_collection("rare_disease_summaries")
            collection.count()
            status["chroma"] = {"ok": True, "detail": "Online"}
        except Exception as exc:
            status["chroma"] = {"ok": False, "detail": str(exc)}

    status["ready"] = status["neo4j"]["ok"] and status["chroma"]["ok"]
    return status


@lru_cache(maxsize=1)
def get_hybrid_retriever():
    return HybridRetriever()

if __name__ == "__main__":
    retriever = HybridRetriever()
    logger.info("Hybrid Retriever initialized and ready.")
