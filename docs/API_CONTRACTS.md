# RareDx API Contracts

This document describes the expected input/output contracts for each major layer in RareDx.

## UI Layer (`app.py`)

### Inputs
- `patient_id`: string, max 50 chars
- `patient_age`: integer, 1-120
- `patient_gender`: string
- `clinical_note`: string, max 5000 chars
- `selected_symptoms`: list of strings
- `selected_genes`: list of strings

### Outputs
- Writes `st.session_state.diagnostic_results` with keys:
  - `final_report`
  - `candidates`
  - `audit_trail`
  - `patient_id`, `patient_age`, `patient_gender`
  - `clinical_note`, `selected_symptoms`, `selected_genes`

## Agent Layer (`core/agent.py`)

### State Contract
The agent workflow expects a state dictionary with the following keys:

- `clinical_note`: str
- `selected_symptoms`: list[str]
- `selected_genes`: list[str]
- `patient_id`: str
- `patient_age`: int
- `patient_gender`: str
- `graph_results`: list[dict]
- `semantic_results`: list[dict]
- `candidates`: list[dict]
- `validated_evidence`: list[str]
- `final_report`: dict
- `audit_trail`: list[dict]

### Node Outputs
- `entity_extractor_node` returns normalized clinician input and initial audit record
- `graph_query_node` returns `graph_results` with `disease`, `orphacode`, `score`, `graph_paths`, `matched_symptoms`, `matched_genes`
- `semantic_search_node` returns `semantic_results` with `disease`, `orphacode`, `score`, `summary`, and citation URLs
- `fusion_node` returns combined `candidates` and drops unsupported ones
- `validation_node` adds ICD-10 mapping and evidence badges
- `report_node` returns `final_report` containing summary and `traceability_map`

## Retriever Layer (`core/retriever.py`)

### `get_hybrid_retriever()`
- Returns an instance of `HybridRetriever`
- May raise `InfrastructureError` if Neo4j or ChromaDB is unavailable

### `query_graph(selected_symptoms, selected_genes, limit=10)`
- Input: lists of symptom strings and gene symbols
- Output: list of dicts with keys:
  - `disease`, `orphacode`, `score`, `graph_paths`, `matched_symptoms`, `matched_genes`

### `query_semantic(clinical_note, limit=5)`
- Input: free-text clinical note string
- Output: list of dicts with keys:
  - `disease`, `orphacode`, `score`, `summary`

### `get_infrastructure_status()`
- Returns a dict with `neo4j`, `chroma`, and `ready` flags

## Reporting Layer (`scripts/report_generator.py`)

### `generate_pdf_report(...)`
- Inputs: patient metadata, selected inputs, candidate list, traceability map, audit trail, summary
- Output: binary PDF bytes suitable for `st.download_button`

### `generate_excel_report(...)`
- Inputs: same structure as PDF generator
- Output: binary XLSX bytes suitable for `st.download_button`

## Error Contract
- `InfrastructureError` is raised for service/connectivity failures
- UI layer should catch `InfrastructureError` and show a user-friendly message
- Internal validation errors should be surfaced only after input validation succeeds
