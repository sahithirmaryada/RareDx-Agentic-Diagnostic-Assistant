import json
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
import logging
import requests
from core.config import config

logger = logging.getLogger(__name__)
NCBI_API_KEY = config.ncbi_api_key
CACHE_FILE = config.resolve_path(config.pubmed_cache_file)
PRODUCT4_XML = config.resolve_path(config.orphanet_product4_xml)
PRODUCT6_XML = config.resolve_path(config.orphanet_product6_xml)


def _normalize_name(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _extract_pmids(text):
    return re.findall(r"(\d+)\s*\[PMID\]", text or "")


def _dedupe(values):
    return list(dict.fromkeys(values))


@lru_cache(maxsize=1)
def _load_orphanet_pubmed_index():
    by_orphacode = {}
    by_name = {}

    context = ET.iterparse(PRODUCT4_XML, events=("end",))
    for _, elem in context:
        if elem.tag != "HPODisorderSetStatus":
            continue

        disorder = elem.find("./Disorder")
        if disorder is not None:
            _add_lookup_entries(
                by_orphacode,
                by_name,
                (disorder.findtext("OrphaCode") or "").strip(),
                (disorder.findtext("Name") or "").strip(),
                [
                    synonym.text.strip()
                    for synonym in disorder.findall("./SynonymList/Synonym")
                    if synonym.text and synonym.text.strip()
                ],
                _extract_pmids(elem.findtext("Source")),
            )

        elem.clear()

    context = ET.iterparse(PRODUCT6_XML, events=("end",))
    for _, elem in context:
        if elem.tag != "Disorder":
            continue

        _add_lookup_entries(
            by_orphacode,
            by_name,
            (elem.findtext("OrphaCode") or "").strip(),
            (elem.findtext("Name") or "").strip(),
            [
                synonym.text.strip()
                for synonym in elem.findall("./SynonymList/Synonym")
                if synonym.text and synonym.text.strip()
            ],
            _dedupe(
                pmid
                for text in elem.itertext()
                if "[PMID]" in (text or "")
                for pmid in _extract_pmids(text)
            ),
        )

        elem.clear()

    return {"by_orphacode": by_orphacode, "by_name": by_name}


def _add_lookup_entries(by_orphacode, by_name, orphacode, disease_name, synonyms, pmids):
    if not pmids:
        return

    unique_pmids = _dedupe(pmids)
    if orphacode:
        by_orphacode[orphacode] = _dedupe(by_orphacode.get(orphacode, []) + unique_pmids)

    for label in [disease_name, *synonyms]:
        normalized = _normalize_name(label)
        if normalized:
            by_name[normalized] = _dedupe(by_name.get(normalized, []) + unique_pmids)


def find_orphanet_citations(query_name=None, orphacode=None, limit=5):
    index = _load_orphanet_pubmed_index()

    pmids = []
    if orphacode:
        pmids = index["by_orphacode"].get(str(orphacode).strip(), [])

    if not pmids and query_name:
        pmids = index["by_name"].get(_normalize_name(query_name), [])

    return [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in pmids[:limit]]


def get_pubmed_ids(disease_query):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": disease_query,
        "retmode": "json",
        "retmax": 5,
    }
    if NCBI_API_KEY and not NCBI_API_KEY.startswith("your_"):
        params["api_key"] = NCBI_API_KEY

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("esearchresult", {}).get("idlist", [])
    except requests.Timeout as e:
        logger.error("PubMed eSearch request timed out", exc_info=True)
        return []
    except requests.RequestException as e:
        logger.error(f"Error during ID search: {e}", exc_info=True)
        return []


def fetch_details(pmid_list):
    if not pmid_list:
        return []
    return [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in pmid_list]


def fetch_pubmed_details(pmid_list):
    """Fetch detailed PubMed article information including titles and abstracts."""
    if not pmid_list:
        return []

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmid_list),
        "retmode": "xml",
        "rettype": "abstract",
    }
    if NCBI_API_KEY and not NCBI_API_KEY.startswith("your_"):
        params["api_key"] = NCBI_API_KEY

    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        articles = []
        for article in root.findall(".//PubmedArticle"):
            medline_citation = article.find(".//MedlineCitation")
            if medline_citation is None:
                continue

            pmid_elem = medline_citation.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None

            article_elem = medline_citation.find(".//Article")
            if article_elem is None:
                continue

            title_elem = article_elem.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "No title available"

            abstract_elem = article_elem.find(".//Abstract")
            abstract = ""
            if abstract_elem is not None:
                abstract_texts = abstract_elem.findall(".//AbstractText")
                abstract = " ".join([text.text for text in abstract_texts if text.text])

            pub_date_elem = article_elem.find(".//PubDate")
            year = ""
            if pub_date_elem is not None:
                year_elem = pub_date_elem.find(".//Year")
                if year_elem is not None:
                    year = year_elem.text

            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract[:500] + "..." if len(abstract) > 500 else abstract,  # Truncate long abstracts
                "year": year,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })

        return articles
    except requests.Timeout:
        logger.error("PubMed eFetch request timed out", exc_info=True)
        return []
    except requests.RequestException as e:
        logger.error(f"Error during PubMed details fetch: {e}", exc_info=True)
        return []


def search_research(disease_query, orphacode=None):
    local_results = find_orphanet_citations(disease_query, orphacode=orphacode)
    if local_results:
        # Convert URLs back to PMIDs and fetch details
        pmids = [url.rstrip('/').split('/')[-1] for url in local_results if url.strip()]
        return fetch_pubmed_details(pmids)

    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            try:
                cache = json.load(f)
                if disease_query in cache:
                    logger.info(f"Returning cached results for {disease_query}...")
                    return cache[disease_query]
            except json.JSONDecodeError:
                logger.warning("PubMed cache file is invalid JSON. Rebuilding cache.")
                cache = {}
    else:
        cache = {}

    logger.info(f"Searching PubMed for {disease_query}...")
    ids = get_pubmed_ids(disease_query)
    results = fetch_pubmed_details(ids)

    if results:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        cache[disease_query] = results
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=4)

    return results


if __name__ == "__main__":
    test_disease = "Pompe disease"
    findings = search_research(test_disease)
    logger.info(f"Evidence for {test_disease}: {findings}")
