import logging
import os
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


PRODUCT1_XML = resolve_path(os.getenv("ORPHANET_PRODUCT1_XML", "data/raw/en_product1.xml"))

RELATION_PRIORITY = {
    "Specific code": 0,
    "Attributed code": 1,
    "Index term": 2,
    "Inclusion term": 3,
}


def _normalize_name(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _relation_rank(relation_name):
    for prefix, rank in RELATION_PRIORITY.items():
        if relation_name.startswith(prefix):
            return rank
    return len(RELATION_PRIORITY)


def _is_better_mapping(candidate, current):
    if current is None:
        return True

    candidate_key = (
        0 if candidate["validation_status"] == "Validated" else 1,
        _relation_rank(candidate["mapping_relation"]),
        candidate["code"],
    )
    current_key = (
        0 if current["validation_status"] == "Validated" else 1,
        _relation_rank(current["mapping_relation"]),
        current["code"],
    )
    return candidate_key < current_key


@lru_cache(maxsize=1)
def _load_icd10_index():
    by_orphacode = {}
    by_name = {}

    context = ET.iterparse(PRODUCT1_XML, events=("end",))
    for _, elem in context:
        if elem.tag != "Disorder":
            continue

        orphacode = (elem.findtext("OrphaCode") or "").strip()
        disease_name = (elem.findtext("Name") or "").strip()
        synonyms = [
            synonym.text.strip()
            for synonym in elem.findall("./SynonymList/Synonym")
            if synonym.text and synonym.text.strip()
        ]

        best_mapping = None
        for ext_ref in elem.findall("./ExternalReferenceList/ExternalReference"):
            source = (ext_ref.findtext("Source") or "").strip()
            code = (ext_ref.findtext("Reference") or "").strip()
            if source != "ICD-10" or not code:
                continue

            mapping = {
                "code": code,
                "disease_name": disease_name,
                "orphacode": orphacode,
                "mapping_relation": (ext_ref.findtext("./DisorderMappingICDRelation/Name") or "").strip(),
                "validation_status": (ext_ref.findtext("./DisorderMappingValidationStatus/Name") or "").strip(),
            }
            if _is_better_mapping(mapping, best_mapping):
                best_mapping = mapping

        if best_mapping:
            by_orphacode[orphacode] = best_mapping
            for label in [disease_name, *synonyms]:
                normalized = _normalize_name(label)
                if not normalized:
                    continue
                if _is_better_mapping(best_mapping, by_name.get(normalized)):
                    by_name[normalized] = best_mapping

        elem.clear()

    return {"by_orphacode": by_orphacode, "by_name": by_name}


def find_icd10_code(query_name=None, orphacode=None):
    index = _load_icd10_index()

    mapping = None
    if orphacode:
        mapping = index["by_orphacode"].get(str(orphacode).strip())

    if mapping is None and query_name:
        mapping = index["by_name"].get(_normalize_name(query_name))

    if mapping is None:
        return {
            "input": query_name,
            "orphacode": orphacode,
            "matched_code": None,
            "official_desc": None,
            "confidence": 0.0,
            "mapping_relation": None,
            "validation_status": None,
        }

    return {
        "input": query_name,
        "orphacode": mapping["orphacode"],
        "matched_code": mapping["code"],
        "official_desc": mapping["disease_name"],
        "confidence": 1.0,
        "mapping_relation": mapping["mapping_relation"],
        "validation_status": mapping["validation_status"],
    }


if __name__ == "__main__":
    logger.info(find_icd10_code("Pompe disease"))
