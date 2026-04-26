import xml.etree.ElementTree as ET
import logging
import chromadb
from sentence_transformers import SentenceTransformer
from core.config import config

logger = logging.getLogger(__name__)

class SemanticIngestor:
    def __init__(self):
        # Initialize BioLORD via sentence-transformers
        logger.info(f"Loading BioLORD model: {config.biolord_model_name}...")
        self.model = SentenceTransformer(config.biolord_model_name)
        
        # Connect to ChromaDB (running in your Docker container)
        self.client = chromadb.HttpClient(host=config.chroma_host, port=config.chroma_port)
        
        # Create or get the collection
        self.collection = self.client.get_or_create_collection(
            name="rare_disease_summaries",
            metadata={"hnsw:space": "cosine"} # Use cosine similarity as per project specs
        )

    def ingest_summaries(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        documents = []
        metadatas = []
        ids = []

        logger.info("Parsing Orphanet disease summaries...")
        for disorder in root.findall(".//Disorder"):
            orphacode = disorder.find("OrphaCode").text
            name = disorder.find("Name").text
            
            # Extract summary/definition if available
            definition = disorder.find(".//Definition")
            if definition is not None and definition.text:
                summary_text = f"Disease: {name}. Definition: {definition.text}"
                
                documents.append(summary_text)
                metadatas.append({"orphacode": orphacode, "source": "Orphanet"})
                ids.append(f"orphacode_{orphacode}")

        # 2. Batch Embedding and Storage
        if documents:
            logger.info(f"Embedding {len(documents)} summaries... this may take a moment.")
            embeddings = self.model.encode(documents)
            
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info("Successfully stored semantic summaries in ChromaDB.")

if __name__ == "__main__":
    ingestor = SemanticIngestor()
    try:
        ingestor.ingest_summaries(config.resolve_path(config.orphanet_product1_xml))
    except FileNotFoundError:
        logger.error(
            f"Error: {config.orphanet_product1_xml} not found. Check ORPHANET_PRODUCT1_XML in .env."
        )
    except ET.ParseError as e:
        logger.error(f"Error parsing Orphanet XML: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
