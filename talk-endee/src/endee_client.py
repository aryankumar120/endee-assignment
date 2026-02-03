import requests
import msgpack
from typing import List, Dict, Any
from src.config import Config

class EndeeClient:
    def __init__(self, host: str = None):
        self.host = host or Config.ENDEE_HOST
        self.base_url = f"{self.host}{Config.ENDEE_API_BASE}"
        print(f"Endee client initialized: {self.base_url}")

    def create_index(self, index_name: str, vector_dim: int) -> Dict[str, Any]:
        url = f"{self.base_url}/index/create"
        payload = {"index_name": index_name, "dim": vector_dim, "space_type": "cosine"}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        # Server may return 200 with empty body; handle gracefully
        if response.text:
            return response.json()
        return {"status": "success", "index": index_name, "dim": vector_dim}

    def upsert_vectors(self, index_name: str, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        url = f"{self.base_url}/index/{index_name}/vector/insert"
        headers = {"Content-Type": "application/json"}
        payload = vectors
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"  HTTP Error: {response.status_code} - {response.text[:200]}")
            raise
        if response.text:
            return response.json()
        else:
            return {"status": "success", "inserted": len(vectors)}

    def search(self, index_name: str, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/index/{index_name}/search"
        headers = {"Content-Type": "application/json"}
        payload = {"k": top_k, "vector": query_vector, "ef": 128}
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '')
        if 'msgpack' in content_type:
            try:
                unpacked = msgpack.unpackb(response.content, raw=False)
                if isinstance(unpacked, list):
                    return unpacked
                elif isinstance(unpacked, dict) and "results" in unpacked:
                    return unpacked["results"]
                else:
                    return []
            except Exception as e:
                print(f"  Failed to decode MessagePack: {e}")
                return []
        if not response.text:
            return []
        try:
            results = response.json()
        except:
            return []
        if isinstance(results, list):
            return results
        elif isinstance(results, dict) and "results" in results:
            return results["results"]
        else:
            return results if results else []

    def delete_vector(self, index_name: str, vector_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/index/{index_name}/delete/{vector_id}"
        response = requests.delete(url)
        response.raise_for_status()
        return response.json()

    def list_indices(self) -> List[str]:
        url = f"{self.base_url}/index/list"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def get_index_info(self, index_name: str) -> Dict[str, Any]:
        url = f"{self.base_url}/index/{index_name}/info"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
