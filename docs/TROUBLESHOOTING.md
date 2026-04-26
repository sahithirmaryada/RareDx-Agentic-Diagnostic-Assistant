# RareDx Troubleshooting Guide

## Common startup issues

### 1. `Neo4j: Offline` in the Streamlit app
- Confirm Docker is running
- Run `docker-compose ps` and verify `raredx-neo4j` is healthy
- Check `.env` values for `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD`
- Verify Neo4j accepts connections on the configured Bolt port

### 2. `ChromaDB: Offline`
- Confirm Docker container `raredx-chroma` is running
- Verify `CHROMA_HOST` and `CHROMA_PORT` match `docker-compose.yml`
- Ensure ChromaDB is reachable from the host

## Data ingestion problems

### XML parse errors
- If `scripts/ingest_semantic.py` fails while parsing Orphanet XML, verify `ORPHANET_PRODUCT1_XML` points to a valid XML file
- Use `python -m xml.etree.ElementTree <file>` to validate syntax

### Missing knowledge graph data
- Run `python scripts/ingest_graph.py` after setting `.env`
- If ingestion fails, check Neo4j credentials and the structure of Orphanet XML files

## Environment validation

### App fails on startup with missing env vars
- The project now uses `core/config.py` with strict validation
- Required variables must be present in `.env`:
  - `GROQ_API_KEY`
  - `NEO4J_URI`
  - `NEO4J_USER`
  - `NEO4J_PASSWORD`

## Dependency issues

### `ModuleNotFoundError`
- Reinstall pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

### Incompatible dependency versions
- Use the pinned versions in `requirements.txt`
- If you need to upgrade, test carefully before deploying

## Debugging tips

- Use logs from the Streamlit console or terminal
- Inspect `traceability_map` and `audit_trail` in the app for evidence state
- When a service is unavailable, `get_infrastructure_status()` reports the exact failure reason

## Contact points

- `app.py`: UI validation and session handling
- `core/retriever.py`: Neo4j and ChromaDB connectivity
- `core/agent.py`: candidate fusion and report assembly
- `mcp/pubmed_tool.py`: PubMed citation lookups and local cache handling
