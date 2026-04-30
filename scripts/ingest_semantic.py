#!/usr/bin/env python3
"""Build the RareDx semantic disease-profile index in ChromaDB."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import config


logger = logging.getLogger(__name__)
COLLECTION_NAME = "rare_disease_summaries"
BATCH_SIZE = 128


@dataclass
class DiseaseProfile:
    orphacode: str
    disease: str
    synonyms: list[str] = field(default_factory=list)
    hpo_terms: list[str] = field(default_factory=list)
    hpo_ids: list[str] = field(default_factory=list)
    frequent_hpo_terms: list[str] = field(default_factory=list)
    genes: list[str] = field(default_factory=list)
    gene_names: list[str] = field(default_factory=list)
    gene_synonyms: list[str] = field(default_factory=list)
    pmids: list[str] = field(default_factory=list)


class SemanticIngestor:
    def __init__(self, reset_collection: bool = False):
        logger.info("Loading embedding model: %s", config.biolord_model_name)
        self.model = SentenceTransformer(config.biolord_model_name)
        self.client = chromadb.HttpClient(host=config.chroma_host, port=config.chroma_port)

        if reset_collection:
            try:
                self.client.delete_collection(COLLECTION_NAME)
                logger.info("Deleted existing Chroma collection: %s", COLLECTION_NAME)
            except Exception:
                logger.info("No existing Chroma collection to delete.")

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_profiles(
        self,
        product4_xml: Path,
        product6_xml: Path,
        product1_xml: Path | None = None,
    ) -> int:
        profiles = build_disease_profiles(product4_xml, product6_xml, product1_xml)
        documents = []
        metadatas = []
        ids = []

        for profile in profiles.values():
            if not profile.hpo_terms and not profile.genes:
                continue
            documents.append(render_profile_document(profile))
            metadatas.append(profile_metadata(profile))
            ids.append(f"orphacode_{profile.orphacode}")

        if not documents:
            raise ValueError(
                "No semantic disease profiles were generated from the Orphanet XML files."
            )

        logger.info("Embedding and upserting %s disease profiles...", len(documents))
        for start in range(0, len(documents), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_documents = documents[start:end]
            embeddings = self.model.encode(batch_documents)
            self.collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings.tolist(),
                documents=batch_documents,
                metadatas=metadatas[start:end],
            )
            logger.info("Upserted profiles %s-%s", start + 1, min(end, len(documents)))

        return len(documents)


def build_disease_profiles(
    product4_xml: Path,
    product6_xml: Path,
    product1_xml: Path | None = None,
) -> dict[str, DiseaseProfile]:
    profiles: dict[str, DiseaseProfile] = {}
    if product1_xml is not None:
        merge_disease_name_profiles(profiles, product1_xml)
    merge_phenotype_profiles(profiles, product4_xml)
    merge_gene_profiles(profiles, product6_xml)
    propagate_related_profile_signal(profiles)
    return profiles


def merge_disease_name_profiles(
    profiles: dict[str, DiseaseProfile], product1_xml: Path
) -> None:
    logger.info("Parsing disease names and synonyms from %s", product1_xml)
    context = ET.iterparse(product1_xml, events=("end",))
    for _, elem in context:
        if elem.tag != "Disorder":
            continue

        orphacode = clean_orphacode(elem.findtext("OrphaCode"))
        disease = clean_text(elem.findtext("Name"))
        if not orphacode or not disease:
            elem.clear()
            continue

        profile = profiles.setdefault(
            orphacode, DiseaseProfile(orphacode=orphacode, disease=disease)
        )
        profile.disease = profile.disease or disease
        profile.synonyms = dedupe(
            profile.synonyms
            + [
                clean_text(synonym.text)
                for synonym in elem.findall("./SynonymList/Synonym")
                if clean_text(synonym.text)
            ]
        )

        elem.clear()


def merge_phenotype_profiles(
    profiles: dict[str, DiseaseProfile], product4_xml: Path
) -> None:
    logger.info("Parsing phenotype profiles from %s", product4_xml)
    context = ET.iterparse(product4_xml, events=("end",))
    for _, elem in context:
        if elem.tag != "HPODisorderSetStatus":
            continue

        disorder = elem.find("./Disorder")
        if disorder is not None:
            orphacode = clean_orphacode(disorder.findtext("OrphaCode"))
            disease = clean_text(disorder.findtext("Name"))
            if orphacode and disease:
                profile = profiles.setdefault(
                    orphacode, DiseaseProfile(orphacode=orphacode, disease=disease)
                )
                profile.disease = profile.disease or disease

                for association in disorder.findall(".//HPODisorderAssociation"):
                    hpo_id = clean_text(association.findtext(".//HPOId"))
                    term = clean_text(association.findtext(".//HPOTerm"))
                    frequency = clean_text(association.findtext(".//HPOFrequency/Name"))
                    if hpo_id:
                        profile.hpo_ids.append(hpo_id)
                    if term:
                        profile.hpo_terms.append(term)
                        if is_frequent_phenotype(frequency):
                            profile.frequent_hpo_terms.append(term)

                profile.hpo_ids = dedupe(profile.hpo_ids)
                profile.hpo_terms = dedupe(profile.hpo_terms)
                profile.frequent_hpo_terms = dedupe(profile.frequent_hpo_terms)

        elem.clear()


def merge_gene_profiles(profiles: dict[str, DiseaseProfile], product6_xml: Path) -> None:
    logger.info("Parsing gene profiles from %s", product6_xml)
    context = ET.iterparse(product6_xml, events=("end",))
    for _, elem in context:
        if elem.tag != "Disorder":
            continue

        orphacode = clean_orphacode(elem.findtext("OrphaCode"))
        disease = clean_text(elem.findtext("Name"))
        if not orphacode or not disease:
            elem.clear()
            continue

        profile = profiles.setdefault(
            orphacode, DiseaseProfile(orphacode=orphacode, disease=disease)
        )
        profile.disease = profile.disease or disease

        for association in elem.findall(".//DisorderGeneAssociation"):
            pmids = extract_pmids(association.findtext("SourceOfValidation"))
            profile.pmids.extend(pmids)

            gene = association.find("./Gene")
            if gene is None:
                continue

            symbol = clean_text(gene.findtext("Symbol")).upper()
            gene_name = clean_text(gene.findtext("Name"))
            synonyms = [
                clean_text(synonym.text)
                for synonym in gene.findall("./SynonymList/Synonym")
                if clean_text(synonym.text)
            ]

            if symbol:
                profile.genes.append(symbol)
            if gene_name:
                profile.gene_names.append(gene_name)
            profile.gene_synonyms.extend(synonyms)

        profile.genes = dedupe(profile.genes)
        profile.gene_names = dedupe(profile.gene_names)
        profile.gene_synonyms = dedupe(profile.gene_synonyms)
        profile.pmids = dedupe(profile.pmids)

        elem.clear()


def propagate_related_profile_signal(profiles: dict[str, DiseaseProfile]) -> None:
    """Roll subtype phenotype/gene signal up to broad parent-like profiles."""
    profile_list = list(profiles.values())
    signal_profiles = [
        profile for profile in profile_list if profile.hpo_terms or profile.genes
    ]

    for parent in profile_list:
        parent_name = normalize_profile_name(parent.disease)
        if len(parent_name) < 8:
            continue

        for child in signal_profiles:
            if child.orphacode == parent.orphacode:
                continue

            child_name = normalize_profile_name(child.disease)
            if not is_related_profile_name(parent_name, child_name):
                continue

            parent.hpo_terms.extend(child.hpo_terms)
            parent.hpo_ids.extend(child.hpo_ids)
            parent.frequent_hpo_terms.extend(child.frequent_hpo_terms)
            parent.genes.extend(child.genes)
            parent.gene_names.extend(child.gene_names)
            parent.gene_synonyms.extend(child.gene_synonyms)
            parent.pmids.extend(child.pmids)

        parent.hpo_terms = dedupe(parent.hpo_terms)
        parent.hpo_ids = dedupe(parent.hpo_ids)
        parent.frequent_hpo_terms = dedupe(parent.frequent_hpo_terms)
        parent.genes = dedupe(parent.genes)
        parent.gene_names = dedupe(parent.gene_names)
        parent.gene_synonyms = dedupe(parent.gene_synonyms)
        parent.pmids = dedupe(parent.pmids)


def is_related_profile_name(parent_name: str, child_name: str) -> bool:
    if parent_name == child_name:
        return False
    if parent_name in child_name:
        return True
    parent_tokens = set(parent_name.split())
    child_tokens = set(child_name.split())
    return len(parent_tokens) >= 2 and parent_tokens.issubset(child_tokens)


def render_profile_document(profile: DiseaseProfile) -> str:
    sections = [
        f"Disease: {profile.disease}.",
        f"ORPHA:{profile.orphacode}.",
    ]

    if profile.synonyms:
        sections.append(f"Synonyms: {join_limited(profile.synonyms, 30)}.")
    if profile.genes:
        sections.append(f"Associated genes: {join_limited(profile.genes, 40)}.")
    if profile.gene_names:
        sections.append(f"Gene names: {join_limited(profile.gene_names, 30)}.")
    if profile.gene_synonyms:
        sections.append(f"Gene aliases: {join_limited(profile.gene_synonyms, 30)}.")
    if profile.frequent_hpo_terms:
        sections.append(
            f"High-frequency phenotypes: {join_limited(profile.frequent_hpo_terms, 60)}."
        )
    if profile.hpo_terms:
        sections.append(f"Phenotypes: {join_limited(profile.hpo_terms, 120)}.")
    if profile.hpo_ids:
        sections.append(f"HPO IDs: {join_limited(profile.hpo_ids, 120)}.")
    if profile.pmids:
        sections.append(f"Source PMIDs: {join_limited(profile.pmids, 20)}.")

    sections.append("Evidence source: Orphanet phenotype and gene annotations.")
    return " ".join(sections)


def profile_metadata(profile: DiseaseProfile) -> dict[str, Any]:
    return {
        "disease": profile.disease,
        "orphacode": profile.orphacode,
        "source": "Orphanet_Product1_Product4_Product6",
        "synonyms_json": json.dumps(profile.synonyms, ensure_ascii=True),
        "hpo_count": len(profile.hpo_terms),
        "gene_count": len(profile.genes),
        "pmid_count": len(profile.pmids),
        "hpo_terms_json": json.dumps(profile.hpo_terms, ensure_ascii=True),
        "hpo_ids_json": json.dumps(profile.hpo_ids, ensure_ascii=True),
        "genes_json": json.dumps(profile.genes, ensure_ascii=True),
        "pmids_json": json.dumps(profile.pmids, ensure_ascii=True),
    }


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def clean_orphacode(value: str | None) -> str:
    return clean_text(value).replace("ORPHA:", "").strip()


def dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def extract_pmids(text: str | None) -> list[str]:
    return dedupe(re.findall(r"(\d+)\s*\[PMID\]", text or ""))


def normalize_profile_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def is_frequent_phenotype(frequency: str) -> bool:
    normalized = frequency.lower()
    return "very frequent" in normalized or normalized.startswith("frequent")


def join_limited(values: list[str], limit: int) -> str:
    values = dedupe(values)
    displayed = values[:limit]
    suffix = f"; +{len(values) - limit} more" if len(values) > limit else ""
    return "; ".join(displayed) + suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build/update the RareDx Chroma semantic disease-profile index."
    )
    parser.add_argument(
        "--product4",
        type=Path,
        default=config.resolve_path(config.orphanet_product4_xml),
        help="Orphanet phenotype XML path.",
    )
    parser.add_argument(
        "--product6",
        type=Path,
        default=config.resolve_path(config.orphanet_product6_xml),
        help="Orphanet gene XML path.",
    )
    parser.add_argument(
        "--product1",
        type=Path,
        default=config.resolve_path(config.orphanet_product1_xml),
        help="Orphanet disease names/synonyms XML path.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the Chroma collection before ingesting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    ingestor = SemanticIngestor(reset_collection=args.reset)
    count = ingestor.ingest_profiles(args.product4, args.product6, args.product1)
    logger.info("Semantic ingestion complete. Indexed %s disease profiles.", count)
