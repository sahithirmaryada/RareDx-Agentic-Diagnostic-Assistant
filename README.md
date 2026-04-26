# RareDx

RareDx is an evidence-grounded diagnostic assistant for rare diseases. It combines structured knowledge from Neo4j, semantic memory from ChromaDB, and curated clinical validation to produce traceable diagnostic candidates.

## Key Features

- Streamlit-based clinician UI
- Hybrid retrieval using Neo4j and ChromaDB
- Evidence enforcement: candidates without graph paths or PubMed citations are dropped
- Audit trail and traceability map for clinical review
- Downloadable PDF and Excel reports
- Strict environment validation via Pydantic

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- `neo4j` and `chromadb` services available via `docker-compose.yml`
- `.env` file with required credentials

## Setup

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` and provide values for the required secrets:

- `GROQ_API_KEY`
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`

3. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Start the knowledge services:

```bash
docker-compose up -d
```

5. Launch the Streamlit app:

```bash
streamlit run app.py
```

## Project Structure

- `app.py` - Streamlit interface and application entry point
- `core/agent.py` - Diagnostic workflow and reasoning state machine
- `core/retriever.py` - Neo4j and ChromaDB hybrid retrieval implementation
- `core/config.py` - Strict environment configuration using Pydantic
- `scripts/` - Data ingestion helpers and report generation utilities
- `mcp/` - Medical citation and ICD-10 mapping helpers
- `tests/` - Unit tests for core workflow
- `docs/` - Architecture, API contracts, and troubleshooting guides

## Configuration

The project uses `core/config.py` to validate required environment variables. No sensitive defaults are allowed for Neo4j credentials or Groq API keys.

Required values in `.env`:

- `GROQ_API_KEY`
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`

Optional values with defaults:

- `CHROMA_HOST`
- `CHROMA_PORT`
- `BIOLORD_MODEL_NAME`

## Troubleshooting

See `docs/TROUBLESHOOTING.md` for common issues when starting services, loading data, or running the app.

## API Contracts

See `docs/API_CONTRACTS.md` for layer contracts between the UI, agent, retriever, and reporting components.
