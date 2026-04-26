from neo4j import GraphDatabase
from core.config import config

def get_all_symptoms():
    """Fetches all unique symptom names from the Neo4j graph."""
    with GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password)) as driver:
        with driver.session() as session:
            # Cypher query to get distinct symptom names [cite: 602, 658]
            query = "MATCH (s:Symptom) RETURN s.name AS name ORDER BY name"
            result = session.run(query)
            # Return as a simple list of strings
            return [record["name"] for record in result if record["name"]]
        
def get_all_genes():
    """Fetches all unique gene symbols from the Neo4j graph."""
    with GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password)) as driver:
        with driver.session() as session:
            # Cypher query to get distinct gene symbols
            query = "MATCH (g:Gene) RETURN g.symbol AS symbol ORDER BY symbol"
            result = session.run(query)
            return [record["symbol"] for record in result if record["symbol"]]
