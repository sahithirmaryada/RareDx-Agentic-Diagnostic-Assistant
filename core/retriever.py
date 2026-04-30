from functools import lru_cache
import logging
import json
from core.config import config

logger = logging.getLogger(__name__)
SEMANTIC_COLLECTION = "rare_disease_summaries"
SEMANTIC_POOL_SIZE = 50
STRUCTURED_TERM_LIMIT = 50

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
            self.collection = self.chroma_client.get_collection(SEMANTIC_COLLECTION)
            if self.collection.count() == 0:
                raise InfrastructureError(
                    "ChromaDB semantic collection is empty. Run scripts/ingest_semantic.py --reset."
                )
        except Exception as exc:
            self.driver.close()
            if isinstance(exc, InfrastructureError):
                raise
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

    def query_semantic(
        self,
        clinical_note,
        selected_symptoms=None,
        selected_genes=None,
        limit=10,
    ):
        query_text = build_semantic_query(
            clinical_note,
            selected_symptoms=selected_symptoms,
            selected_genes=selected_genes,
        )
        if not query_text:
            return []

        query_embedding = self.model.encode([query_text])
        pool_limit = max(limit, SEMANTIC_POOL_SIZE)

        try:
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=pool_limit,
                include=["documents", "metadatas", "distances"],
            )
        except ChromaException as exc:
            raise InfrastructureError(
                "ChromaDB query failed. Grounded diagnostic services cannot run."
            ) from exc

        candidate_map = {}
        merge_raw_semantic_results(
            candidate_map,
            documents=results.get("documents", [[]])[0],
            metadatas=results.get("metadatas", [[]])[0],
            distances=results.get("distances", [[]])[0],
            selected_symptoms=selected_symptoms or [],
            selected_genes=selected_genes or [],
            source="vector",
        )

        structured_results = self.query_structured_semantic_candidates(
            selected_symptoms or [],
            selected_genes or [],
        )
        merge_raw_semantic_results(
            candidate_map,
            documents=structured_results["documents"],
            metadatas=structured_results["metadatas"],
            distances=[None] * len(structured_results["documents"]),
            selected_symptoms=selected_symptoms or [],
            selected_genes=selected_genes or [],
            source="structured",
        )

        semantic_results = sorted(
            candidate_map.values(),
            key=lambda candidate: candidate["score"],
            reverse=True,
        )
        return semantic_results[:limit]

    def query_structured_semantic_candidates(self, selected_symptoms, selected_genes):
        """Pull exact gene/HPO profile matches from Chroma metadata documents."""
        documents = []
        metadatas = []
        seen = set()
        terms = [
            *[str(gene).strip().upper() for gene in selected_genes if str(gene).strip()],
            *[str(symptom).strip() for symptom in selected_symptoms if str(symptom).strip()],
        ]

        for term in dict.fromkeys(terms):
            try:
                results = self.collection.get(
                    where_document={"$contains": term},
                    limit=STRUCTURED_TERM_LIMIT,
                    include=["documents", "metadatas"],
                )
            except ChromaException:
                logger.debug("Structured Chroma lookup failed for term: %s", term, exc_info=True)
                continue

            for document, metadata in zip(
                results.get("documents", []),
                results.get("metadatas", []),
            ):
                key = semantic_identity_key(metadata or {}, document)
                if not key or key in seen:
                    continue
                seen.add(key)
                documents.append(document)
                metadatas.append(metadata or {})

        return {"documents": documents, "metadatas": metadatas}


def build_semantic_query(clinical_note, selected_symptoms=None, selected_genes=None):
    parts = []
    note = (clinical_note or "").strip()
    symptoms = [str(item).strip() for item in (selected_symptoms or []) if str(item).strip()]
    genes = [str(item).strip().upper() for item in (selected_genes or []) if str(item).strip()]

    if symptoms:
        parts.append(f"Phenotypes: {'; '.join(dict.fromkeys(symptoms))}.")
    if genes:
        parts.append(f"Genes: {'; '.join(dict.fromkeys(genes))}.")
    if note:
        parts.append(f"Clinical note: {note}")

    return " ".join(parts).strip()


def merge_raw_semantic_results(
    candidate_map,
    documents,
    metadatas,
    distances,
    selected_symptoms,
    selected_genes,
    source,
):
    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        candidate = semantic_candidate_from_raw(
            document=document,
            metadata=metadata or {},
            distance=distance,
            rank=rank,
            selected_symptoms=selected_symptoms,
            selected_genes=selected_genes,
            source=source,
        )
        if not candidate:
            continue

        key = semantic_identity_key(metadata or {}, document)
        if not key:
            continue

        existing = candidate_map.get(key)
        if existing is None or candidate["score"] > existing["score"]:
            candidate_map[key] = candidate


def semantic_candidate_from_raw(
    document,
    metadata,
    distance,
    rank,
    selected_symptoms,
    selected_genes,
    source,
):
    disease_name = metadata.get("disease") or _extract_disease_name(document)
    if not disease_name:
        return None

    indexed_symptoms = parse_metadata_list(metadata.get("hpo_terms_json"))
    indexed_genes = parse_metadata_list(metadata.get("genes_json"))
    matched_symptoms = matched_terms(selected_symptoms, indexed_symptoms)
    matched_genes = matched_terms(selected_genes, indexed_genes, normalize_upper=True)
    vector_score = (
        1.0 / (1.0 + max(distance, 0.0))
        if distance is not None
        else 0.0
    )
    score = semantic_rerank_score(
        vector_score=vector_score,
        vector_rank=rank if source == "vector" else None,
        selected_symptoms=selected_symptoms,
        selected_genes=selected_genes,
        matched_symptoms=matched_symptoms,
        matched_genes=matched_genes,
    )

    return {
        "disease": disease_name,
        "orphacode": normalize_orphacode(metadata.get("orphacode")),
        "score": score,
        "vector_score": vector_score,
        "summary": document,
        "matched_symptoms": matched_symptoms,
        "matched_genes": matched_genes,
        "source_pmids": parse_metadata_list(metadata.get("pmids_json")),
    }


def semantic_rerank_score(
    vector_score,
    vector_rank,
    selected_symptoms,
    selected_genes,
    matched_symptoms,
    matched_genes,
):
    symptom_score = rate(len(matched_symptoms), len(selected_symptoms))
    gene_score = 1.0 if matched_genes else 0.0
    vector_rank_score = 1.0 / vector_rank if vector_rank else 0.0
    exact_gene_bonus = 0.20 if matched_genes else 0.0
    exact_symptom_bonus = 0.10 if len(matched_symptoms) >= 2 else 0.0
    phenotype_completeness_bonus = 0.25 if symptom_score >= 0.8 else 0.0
    distractor_gene_penalty = (
        0.08 * (len(matched_genes) - 1)
        if len(selected_genes) >= 3 and len(matched_genes) > 1
        else 0.0
    )

    return (
        (0.20 * vector_score)
        + (0.08 * vector_rank_score)
        + (0.70 * gene_score)
        + (0.95 * symptom_score)
        + exact_gene_bonus
        + exact_symptom_bonus
        + phenotype_completeness_bonus
        - distractor_gene_penalty
    )


def rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def semantic_identity_key(metadata, document):
    orphacode = normalize_orphacode((metadata or {}).get("orphacode"))
    if orphacode:
        return f"orpha:{orphacode}"
    disease_name = (metadata or {}).get("disease") or _extract_disease_name(document)
    if disease_name:
        return f"name:{disease_name.lower()}"
    return ""


def _extract_disease_name(document):
    prefix = "Disease:"
    divider = ". Definition:"
    if not document or not document.startswith(prefix):
        return None
    if divider in document:
        return document[len(prefix):document.index(divider)].strip()
    first_sentence = document.find(".")
    if first_sentence != -1:
        return document[len(prefix):first_sentence].strip()
    return document[len(prefix):].strip()


def normalize_orphacode(orphacode):
    return str(orphacode or "").replace("ORPHA:", "").strip()


def parse_metadata_list(value):
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if item]


def matched_terms(query_terms, indexed_terms, normalize_upper=False):
    if normalize_upper:
        indexed_lookup = {str(item).upper(): item for item in indexed_terms}
        return [
            indexed_lookup[str(term).upper()]
            for term in query_terms
            if str(term).upper() in indexed_lookup
        ]

    indexed_lookup = {str(item).lower(): item for item in indexed_terms}
    return [
        indexed_lookup[str(term).lower()]
        for term in query_terms
        if str(term).lower() in indexed_lookup
    ]


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
