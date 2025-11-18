import os
import glob
import yaml
import json
import numpy as np

from model_loader import load_embedding_model
from embedding_extractor import EmbeddingExtractor
from verifier import EmbeddingVerifier

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    # Carpeta de imágenes detectadas
    frames_dir = "C:/Dansware_dev/AutoCadena/data"
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(frames_dir, ext)))

    if not image_paths:
        print("❌ No se encontraron imágenes en:", frames_dir)
        return

    cfg = load_config()
    dist_dir = cfg["dist_dir"]
    artifacts = cfg["artifacts"]

    # Crear directorio testing
    testing_dir = os.path.join(dist_dir, "testing")
    os.makedirs(testing_dir, exist_ok=True)
    output_file = os.path.join(testing_dir, "detecciones.txt")

    # Cargar modelo y extractor
    model = load_embedding_model(cfg["embedding_model"])
    extractor = EmbeddingExtractor(
        model=model,
        image_size=int(cfg["image_size"]),
        mean=cfg["normalize_mean"],
        std=cfg["normalize_std"]
    )

    # Cargar verificador
    verifier = EmbeddingVerifier(
        faiss_index_path=os.path.join(dist_dir, artifacts["faiss_index"]),
        labels_json_path=os.path.join(dist_dir, artifacts["labels_json"]),
        embedding_dim=int(cfg["embedding_dim"]),
        top_k=int(cfg["top_k"]),
        similarity_threshold=float(cfg["similarity_threshold"]),
        min_matches=int(cfg["min_matches"])
    )

    # Procesar todas las imágenes
    with open(output_file, "w", encoding="utf-8") as f_out:
        for image_path in image_paths:
            emb = extractor.extract_from_path(image_path)
            result = verifier.detailed(emb)

            print("\n🔍 Verificando:", image_path)
            print("✅ Coincidencia:", result["passed"])
            print("📊 Top-K resultados:")
            for label, score in result["top_k"]:
                print(f"  - {score:.3f} :: {label}")

            # Guardar SOLO si la coincidencia es de clase 'cadena'
            save_line = None
            for label, score in result["top_k"]:
                if label == "cadena" and score >= 0.8:
                    save_line = f"{os.path.basename(image_path)} :: {score:.3f} (CADENA >=0.8)"
                    break

            if result["passed"] and not save_line:
                # Coincidencia True pero sin scores >=0.8
                # Solo registrar si los matches son de 'cadena'
                cadena_scores = [f"{s:.3f}" for lab, s in result["top_k"] if lab == "cadena"]
                if cadena_scores:
                    scores_str = ", ".join(cadena_scores)
                    save_line = f"{os.path.basename(image_path)} :: PASSED (CADENA) :: scores [{scores_str}]"

            if save_line:
                f_out.write(save_line + "\n")

    print(f"\n📁 Resultados guardados en: {output_file}")

if __name__ == "__main__":
    main()
