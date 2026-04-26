"""
Configuration management for RareDx using Pydantic for validation.
All environment variables are required - no defaults for sensitive data.
"""
from pathlib import Path
from pydantic import validator
from pydantic_settings import BaseSettings


class RareDxConfig(BaseSettings):
    """Strict configuration with no defaults for sensitive values."""

    # LLM Orchestration
    groq_api_key: str

    # Structured Knowledge (Neo4j) - Required, no defaults
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_http_port: int = 7474
    neo4j_bolt_port: int = 7687

    # Semantic Search
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    biolord_model_name: str = "FremyCompany/BioLORD-2023"

    # Data Files
    orphanet_product1_xml: str = "data/raw/en_product1.xml"
    orphanet_product4_xml: str = "data/raw/en_product4.xml"
    orphanet_product6_xml: str = "data/raw/en_product6.xml"
    pubmed_cache_file: str = "data/processed/pubmed_cache.json"

    # External Tools
    ncbi_api_key: str = ""  # Optional for now

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("neo4j_uri")
    def validate_neo4j_uri(cls, v):
        if not v.startswith(("bolt://", "neo4j://")):
            raise ValueError("NEO4J_URI must start with bolt:// or neo4j://")
        return v

    @validator("groq_api_key", "neo4j_user", "neo4j_password")
    def validate_required_strings(cls, v):
        if not v or not v.strip():
            raise ValueError("This field cannot be empty")
        return v.strip()

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).resolve().parents[1]

    def resolve_path(self, path_value: str) -> Path:
        """Resolve a path relative to project root if not absolute."""
        path = Path(path_value)
        return path if path.is_absolute() else self.project_root / path


# Global config instance
config = RareDxConfig()