import os
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Annotated
import operator
from datetime import datetime
import re
from core.config import config
from groq import Groq
from langgraph.graph import StateGraph, END
from core.retriever import get_hybrid_retriever
from mcp.pubmed_tool import search_research
from mcp.icd10_helper import find_icd10_code

# Load credentials
client = Groq(api_key=config.groq_api_key) if config.groq_api_key else None

# --- STEP 1: Define the Shared State ---
class AgentState(TypedDict):
    clinical_note: str
    selected_symptoms: List[str]
    selected_genes: List[str]
    patient_id: str
    patient_age: int
    patient_gender: str
    graph_results: List[Dict[str, Any]]
    semantic_results: List[Dict[str, Any]]
    candidates: List[Dict[str, Any]]
    validated_evidence: List[str]
    final_report: Dict[str, Any]
    audit_trail: Annotated[List[dict], operator.add] 

# --- STEP 2: Define Reasoning Nodes ---

def entity_extractor_node(state: AgentState):
    """Normalizes clinician-provided structured inputs without replacing them."""
    selected_symptoms = list(dict.fromkeys(state.get("selected_symptoms", [])))
    selected_genes = list(
        dict.fromkeys(gene.strip().upper() for gene in state.get("selected_genes", []) if gene)
    )
    clinical_note = state.get("clinical_note", "").strip()
    patient_id = state.get("patient_id", "Unknown")
    patient_age = state.get("patient_age", 0)
    patient_gender = state.get("patient_gender", "Unknown")

    log = {
        "timestamp": str(datetime.now()),
        "node": "InputNormalizer",
        "action": "Clinician Input Preserved",
        "details": f"Patient {patient_id} (Age {patient_age}, {patient_gender}) | Using {len(selected_symptoms)} symptoms, {len(selected_genes)} genes, and {len(clinical_note)} note characters."
    }
    return {
        "clinical_note": clinical_note,
        "selected_symptoms": selected_symptoms,
        "selected_genes": selected_genes,
        "patient_id": patient_id,
        "patient_age": patient_age,
        "patient_gender": patient_gender,
        "audit_trail": [log]
    }

def graph_query_node(state: AgentState):
    """Grounded Neo4j retrieval from clinician-selected symptoms and genes."""
    retriever = get_hybrid_retriever()
    found = retriever.query_graph(
        state.get("selected_symptoms", []),
        state.get("selected_genes", []),
    )

    log = {
        "timestamp": str(datetime.now()),
        "node": "GraphQuery",
        "action": "Neo4j Traversal",
        "details": f"Found {len(found)} results.",
        "evidence_path": found[0]["graph_paths"][0] if found and found[0]["graph_paths"] else "None"
    }
    return {"graph_results": found, "audit_trail": [log]}

def semantic_search_node(state: AgentState):
    """ChromaDB semantic retrieval from the free-text note, grounded by PubMed citations."""
    retriever = get_hybrid_retriever()
    found = retriever.query_semantic(state.get("clinical_note", ""))
    for result in found:
        result["citations"] = resolve_citations(result.get("disease"), result.get("orphacode"))

    log = {
        "timestamp": str(datetime.now()),
        "node": "SemanticSearch",
        "action": "Vector Retrieval",
        "details": f"Retrieved {len(found)} semantic candidates from clinician notes.",
        "pmid": found[0]["citations"][0] if found and found[0]["citations"] else "None"
    }
    return {"semantic_results": found, "audit_trail": [log]}

def fusion_node(state: AgentState):
    """
    STRICT GUARD: Applies 60/40 Weighted Fusion.
    Any candidate without a Path or PMID is dropped immediately.
    """
    candidate_map = {}
    
    for res in state.get("graph_results", []):
        key = candidate_key(res)
        candidate_map[key] = {
            "disease": res["disease"],
            "orphacode": res.get("orphacode"),
            "score": res["score"] * 0.6,
            "graph_paths": res.get("graph_paths", []),
            "citations": [],
            "matched_symptoms": res.get("matched_symptoms", []),
            "matched_genes": res.get("matched_genes", []),
            "summary": None,
        }

    for res in state.get("semantic_results", []):
        key = candidate_key(res)
        if key in candidate_map:
            candidate_map[key]["score"] += res["score"] * 0.4
            if not candidate_map[key].get("orphacode"):
                candidate_map[key]["orphacode"] = res.get("orphacode")
            if not candidate_map[key].get("disease"):
                candidate_map[key]["disease"] = res["disease"]
            candidate_map[key]["citations"] = res.get("citations", [])
            candidate_map[key]["summary"] = res.get("summary")
        else:
            candidate_map[key] = {
                "disease": res["disease"],
                "orphacode": res.get("orphacode"),
                "score": res["score"] * 0.4,
                "graph_paths": [],
                "citations": res.get("citations", []),
                "matched_symptoms": [],
                "matched_genes": [],
                "summary": res.get("summary"),
            }

    # Ensure graph-only candidates also get PubMed citations if available
    graph_only_citation_count = 0
    for candidate in candidate_map.values():
        if not candidate["citations"] and candidate.get("disease"):
            candidate["citations"] = resolve_citations(
                candidate["disease"], candidate.get("orphacode")
            )
            if candidate["citations"] and candidate["graph_paths"]:
                graph_only_citation_count += 1

    verified = [
        candidate
        for candidate in candidate_map.values()
        if candidate["graph_paths"] or candidate["citations"]
    ]
    verified = sorted(verified, key=lambda x: x["score"], reverse=True)

    log = {
        "timestamp": str(datetime.now()),
        "node": "FusionGuard",
        "action": "60/40 Weighted Fusion Applied",
        "details": (
            f"Verified {len(verified)} candidates. Dropped {len(candidate_map)-len(verified)}. "
            f"Resolved PubMed citations for {graph_only_citation_count} graph-only candidate(s)."
        ),
    }
    return {"candidates": verified, "audit_trail": [log]}

def validation_node(state: AgentState):
    """Adds lightweight grounded metadata without inventing support."""
    validated_candidates = []
    mcp_results = []

    for candidate in state.get("candidates", []):
        enriched_candidate = dict(candidate)
        icd10_mapping = resolve_icd10_code(
            enriched_candidate.get("disease"),
            enriched_candidate.get("orphacode"),
        )
        icd10_code = icd10_mapping.get("matched_code")
        evidence_badges = []

        if enriched_candidate["graph_paths"]:
            evidence_badges.append("Neo4j Path Found")
        if enriched_candidate["citations"]:
            evidence_badges.append("PubMed Grounded")
        if icd10_code:
            evidence_badges.append("ICD-10 Mapped")

        enriched_candidate["icd10"] = icd10_code
        enriched_candidate["evidence_badges"] = evidence_badges
        enriched_candidate["icd10_status"] = (
            f"Mapped via Orphanet ICD-10 ({icd10_mapping.get('mapping_relation')})"
            if icd10_code
            else "No local ICD-10 mapping found"
        )
        enriched_candidate["citation_status"] = (
            f"{len(enriched_candidate['citations'])} PubMed articles found"
            if enriched_candidate["citations"]
            else "No PubMed articles found"
        )
        validated_candidates.append(enriched_candidate)

    if validated_candidates:
        top_candidate = validated_candidates[0]
        if top_candidate["graph_paths"]:
            mcp_results.append(f"Neo4j paths: {len(top_candidate['graph_paths'])}")
        mcp_results.append(top_candidate["citation_status"])
        if top_candidate.get("icd10"):
            mcp_results.append(f"ICD-10 mapped: {top_candidate['icd10']}")
        else:
            mcp_results.append(top_candidate["icd10_status"])

    log = {
        "timestamp": str(datetime.now()),
        "node": "ValidationNode",
        "action": "Grounded Metadata Check",
        "details": f"Validated {len(validated_candidates)} candidates against available evidence.",
        "mcp_output": mcp_results
    }
    return {"candidates": validated_candidates, "validated_evidence": mcp_results, "audit_trail": [log]}

def report_node(state: AgentState):
    """Assembles the final Glass Box report."""
    empty_traceability_map = {
        "graph_paths": [],
        "citations": [],
        "mcp_validation": [],
    }

    if not state["candidates"]:
        return {
            "final_report": {
                "summary": "No grounded candidate met the RareDx evidence requirements.",
                "traceability_map": empty_traceability_map,
            }
        }

    top = state["candidates"][0]
    
    traceability_map = {
        "primary_match": top["disease"],
        "confidence_score": f"{top['score']:.2f}",
        "icd10": top.get("icd10"),
        "graph_paths": top.get("graph_paths", []),
        "citations": [article["pmid"] for article in top.get("citations", []) if article.get("pmid")],
        "mcp_validation": state.get("validated_evidence", []),
    }
    
    return {
        "final_report": {
            "summary": f"Grounded analysis ranked {top['disease']} highest from the verified clinician inputs.",
            "traceability_map": traceability_map
        }
    }


def resolve_citations(disease_name, orphacode=None):
    articles = search_research(disease_name, orphacode=orphacode)
    return articles


def resolve_icd10_code(disease_name, orphacode=None):
    return find_icd10_code(query_name=disease_name, orphacode=orphacode)


def candidate_key(candidate):
    orphacode = (candidate.get("orphacode") or "").strip()
    if orphacode:
        return f"orpha:{orphacode}"
    return f"name:{normalize_disease_name(candidate.get('disease'))}"


def normalize_disease_name(name):
    return re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()

# --- STEP 3: Build the State Machine ---
workflow = StateGraph(AgentState)

workflow.add_node("extractor", entity_extractor_node)
workflow.add_node("graph_query", graph_query_node)
workflow.add_node("semantic_search", semantic_search_node)
workflow.add_node("fusion_guard", fusion_node)
workflow.add_node("validate", validation_node)
workflow.add_node("write_report", report_node)

workflow.set_entry_point("extractor")

workflow.add_edge("extractor", "graph_query")
workflow.add_edge("graph_query", "semantic_search")
workflow.add_edge("semantic_search", "fusion_guard")
workflow.add_edge("fusion_guard", "validate")
workflow.add_edge("validate", "write_report")
workflow.add_edge("write_report", END)

rare_dx_agent = workflow.compile()
