import os
import json
import uuid
from typing import List, Dict, Any
from src.embeddings import EmbeddingService
from src.endee_client import EndeeClient
from src.config import Config

METADATA_STORE_FILE = "vector_metadata.json"

def load_metadata_store() -> Dict[str, Dict[str, Any]]:
    if os.path.exists(METADATA_STORE_FILE):
        with open(METADATA_STORE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_metadata_store(store: Dict[str, Dict[str, Any]]):
    with open(METADATA_STORE_FILE, 'w') as f:
        json.dump(store, f, indent=2)

class DocumentProcessor:
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
    
    def load_text_file(self, file_path: str) -> str:
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def chunk_text(self, text: str) -> List[str]:
        
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a document into chunks
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of chunks with metadata
        """
        text = self.load_text_file(file_path)
        chunks = self.chunk_text(text)
        
        doc_name = os.path.basename(file_path)
        processed_chunks = []
        
        for idx, chunk in enumerate(chunks):
            processed_chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "source": doc_name,
                "chunk_index": idx,
                "total_chunks": len(chunks)
            })
        
        return processed_chunks


class IngestionPipeline:
    
    def __init__(self, index_name: str = "talk_endee"):
        
        self.index_name = index_name
        self.embedding_service = EmbeddingService()
        self.endee_client = EndeeClient()
        self.processor = DocumentProcessor()
        self.metadata_store = load_metadata_store()
        
        self._ensure_index_exists()
    
    def _ensure_index_exists(self):
        
        try:
            indices = self.endee_client.list_indices()
            if self.index_name in indices:
                print(f"Index '{self.index_name}' already exists")
                return
        except Exception as e:
            print(f"Note: Could not check existing indices: {e}")
        
        embedding_dim = self.embedding_service.get_dimension()
        print(f"Creating index '{self.index_name}' with dimension {embedding_dim}")
        try:
            self.endee_client.create_index(self.index_name, embedding_dim)
        except Exception as e:
            # If index already exists (HTTP 409), treat as non-fatal
            resp = getattr(e, 'response', None)
            status = getattr(resp, 'status_code', None)
            if status == 409:
                print(f"Index '{self.index_name}' already exists (409). Continuing...")
                return
            raise
    
    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"Processing document: {file_path}")
        chunks = self.processor.process_document(file_path)
        print(f"Generated {len(chunks)} chunks")
        
        print("Generating embeddings...")
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_service.embed_batch(texts)
        
        
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": chunk["id"],
                "vector": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"]
                }
            })
            self.metadata_store[chunk["id"]] = {
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"]
            }
        
        print(f"Upserting {len(vectors)} vectors to Endee...")
        result = self.endee_client.upsert_vectors(self.index_name, vectors)
        print(f"Upsert result: {result}")
        save_metadata_store(self.metadata_store)
        
        return {
            "file": file_path,
            "chunks": len(chunks),
            "vectors_stored": len(vectors),
            "status": "success"
        }
    
    def ingest_directory(self, directory: str, file_extension: str = ".txt") -> List[Dict[str, Any]]:
        
        results = []
        for file_name in os.listdir(directory):
            if file_name.endswith(file_extension):
                file_path = os.path.join(directory, file_name)
                try:
                    result = self.ingest_file(file_path)
                    results.append(result)
                except Exception as e:
                    print(f"Error ingesting {file_path}: {e}")
                    results.append({
                        "file": file_path,
                        "status": "error",
                        "error": str(e)
                    })
        
        return results
