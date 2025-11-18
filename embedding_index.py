import faiss
import numpy as np
from typing import List, Tuple

class EmbeddingIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner Product (cosine si embeddings normalizados)
        self.labels: List[str] = []

    def add(self, embeddings: np.ndarray, labels: List[str]):
        assert embeddings.dtype == np.float32, "Embeddings deben ser float32"
        if len(embeddings) != len(labels):
            raise ValueError("Número de embeddings y labels no coincide")
        self.index.add(embeddings)
        self.labels.extend(labels)

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        query = query.astype('float32')[None, :]
        D, I = self.index.search(query, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.labels):
                label = "UNKNOWN"
            else:
                label = self.labels[idx]
            results.append((label, float(score)))
        return results

    def save(self, index_path: str, labels_path: str):
        faiss.write_index(self.index, index_path)
        # Guardamos las etiquetas asociadas
        import json
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(index_path: str, labels_path: str, dim: int):
        idx = EmbeddingIndex(dim)
        idx.index = faiss.read_index(index_path)
        import json
        with open(labels_path, "r", encoding="utf-8") as f:
            idx.labels = json.load(f)
        return idx
