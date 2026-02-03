"""
Retriever module - Search Endee and retrieve relevant context
"""

from typing import List, Dict, Any
import json
import os
from src.embeddings import EmbeddingService
from src.endee_client import EndeeClient
from src.config import Config

# Metadata store file location
METADATA_STORE_FILE = "vector_metadata.json"

def load_metadata_store() -> Dict[str, Dict[str, Any]]:
    if os.path.exists(METADATA_STORE_FILE):
        with open(METADATA_STORE_FILE, 'r') as f:
            return json.load(f)
    return {}

class RAGRetriever:
    """Retrieve context from Endee for RAG"""
    
    def __init__(self, index_name: str = "talk_endee"):
        """
        Initialize RAG retriever
        
        Args:
            index_name: Name of the Endee index
        """
        self.index_name = index_name
        self.embedding_service = EmbeddingService()
        self.endee_client = EndeeClient()
        self.metadata_store = load_metadata_store()
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            top_k: Number of results to retrieve (default: from config)
            
        Returns:
            List of retrieved documents with scores
        """
        top_k = top_k or Config.TOP_K
        
        print(f"Embedding query: {query}")
        query_embedding = self.embedding_service.embed(query)
        
        print(f"Searching Endee for top {top_k} results...")
        results = self.endee_client.search(self.index_name, query_embedding, top_k)
        
        retrieved_docs = []
        for idx, result in enumerate(results, 1):
            if isinstance(result, (list, tuple)) and len(result) >= 2:
                distance = result[0]
                vector_id = result[1]
                metadata = self.metadata_store.get(vector_id, {})
                doc = {
                    "rank": idx,
                    "id": vector_id,
                    "score": distance,
                    "text": metadata.get("text", ""),
                    "source": metadata.get("source", "unknown")
                }
            elif isinstance(result, dict):
                doc = {
                    "rank": idx,
                    "id": result.get("id"),
                    "score": result.get("score", 0),
                    "text": result.get("text", ""),
                    "source": result.get("source", "unknown")
                }
            else:
                doc = {
                    "rank": idx,
                    "id": str(idx),
                    "score": 0,
                    "text": str(result),
                    "source": "unknown"
                }
            
            retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for doc in documents:
            context_parts.append(f"[Source: {doc['source']} | Score: {doc['score']:.2f}]\n{doc['text']}")
        
        return "\n\n".join(context_parts)
