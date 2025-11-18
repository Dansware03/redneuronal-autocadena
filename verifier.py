import numpy as np
from embedding_index import EmbeddingIndex

class EmbeddingVerifier:
    def __init__(self, faiss_index_path: str, labels_json_path: str, embedding_dim: int,
                 top_k: int, similarity_threshold: float, min_matches: int):
        self.index = EmbeddingIndex.load(faiss_index_path, labels_json_path, embedding_dim)
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.min_matches = min_matches

    def verify_vector(self, embedding: np.ndarray) -> bool:
        results = self.index.search(embedding.astype('float32'), k=self.top_k)
        # Coincidencia válida si hay suficientes matches de clase "cadena"
        matches = [r for r in results if r[0] == "cadena" and r[1] >= self.similarity_threshold]
        return len(matches) >= self.min_matches

    def detailed(self, embedding: np.ndarray):
        results = self.index.search(embedding.astype('float32'), k=self.top_k)
        passed = sum(1 for label, score in results if label == "cadena" and score >= self.similarity_threshold) >= self.min_matches
        return {
            "top_k": results,
            "passed": passed
        }
