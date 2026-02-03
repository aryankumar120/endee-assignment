from typing import List
from src.config import Config

try:
    from sentence_transformers import SentenceTransformer 
    _HAS_ST = True
except Exception:
    _HAS_ST = False

import hashlib
import struct


class EmbeddingService:

    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.EMBEDDING_MODEL
        if _HAS_ST:
            self._model = SentenceTransformer(self.model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
        else:
            # fallback dimension (matches many small models)
            self._dim = int(Config.EMBEDDING_MODEL.split("/")[-1].count("-") * 8) or 384

    def embed(self, text: str) -> List[float]:
        if _HAS_ST:
            emb = self._model.encode(text, convert_to_tensor=False)
            return emb.tolist()

        h = hashlib.sha256(text.encode("utf-8")).digest()
        vals = []
        while len(vals) < self._dim:
            for i in range(0, len(h), 8):
                if len(vals) >= self._dim:
                    break
                chunk = h[i:i+8]
                if len(chunk) < 8:
                    chunk = chunk.ljust(8, b"\0")
                v = struct.unpack("!Q", chunk)[0]
                vals.append(((v % 1000000) / 1000000.0) * 2.0 - 1.0)
            h = hashlib.sha256(h).digest()
        return vals[:self._dim]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    def get_dimension(self) -> int:
        return self._dim
