import os
import glob
import json
import time
import yaml
import random
import numpy as np

from model_loader import load_embedding_model
from embedding_extractor import EmbeddingExtractor
from embedding_index import EmbeddingIndex
from verifier import EmbeddingVerifier

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def collect_images_from_class(root_dir, class_name):
    """Recorre una carpeta y devuelve lista de (ruta, clase)."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(root_dir, ext)))
    return [(p, class_name) for p in paths]

def sample_images(dir_path, n):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    all_paths = []
    for ext in exts:
        all_paths.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
    return random.sample(all_paths, min(n, len(all_paths)))

def test_batch(paths, expected_pass, extractor, verifier):
    errors = 0
    failed_paths = []
    for p in paths:
        emb = extractor.extract_from_path(p)
        result = verifier.verify_vector(emb)
        if result != expected_pass:
            errors += 1
            failed_paths.append(p)
    return errors, len(paths), failed_paths

def run_validation_tests(cfg, extractor, verifier):
    print("Ejecutando validación extendida...")

    sample_size = int(cfg["sanity_check"]["sample_queries"])
    error_threshold = 0.30  # 30%
    dist_dir = cfg["dist_dir"]
    fallos_path = os.path.join(dist_dir, "fallos_validacion.txt")

    pos_paths = sample_images(cfg["cadena_dir"], sample_size)
    neg_paths = sample_images(cfg["normal_dir"], sample_size)
    hard_dir = os.path.join(cfg["normal_dir"], "paisajes_dificiles")
    hard_paths = sample_images(hard_dir, sample_size)

    pos_errors, pos_total, pos_failed = test_batch(pos_paths, True, extractor, verifier)
    neg_errors, neg_total, neg_failed = test_batch(neg_paths, False, extractor, verifier)
    hard_errors, hard_total, hard_failed = test_batch(hard_paths, False, extractor, verifier)

    pos_rate = pos_errors / pos_total if pos_total else 0
    neg_rate = neg_errors / neg_total if neg_total else 0
    hard_rate = hard_errors / hard_total if hard_total else 0

    print(f"Errores aceptación (cadena): {pos_errors}/{pos_total} ({pos_rate:.2%})")
    print(f"Errores rechazo (normal): {neg_errors}/{neg_total} ({neg_rate:.2%})")
    print(f"Errores difíciles (paisajes): {hard_errors}/{hard_total} ({hard_rate:.2%})")

    fallos_detectados = any([
        pos_rate > error_threshold,
        neg_rate > error_threshold,
        hard_rate > error_threshold
    ])

    with open(fallos_path, "w", encoding="utf-8") as f:
        f.write("📋 Fallos de validación extendida\n\n")

        if pos_failed:
            f.write("⚠️ Imágenes de 'cadena' que fallaron:\n")
            for p in pos_failed:
                f.write(f"{p}\n")
            f.write("\n")

        if neg_failed:
            f.write("⚠️ Imágenes de 'normal' aceptadas erróneamente:\n")
            for p in neg_failed:
                f.write(f"{p}\n")
            f.write("\n")

        if hard_failed:
            f.write("⚠️ Imágenes de 'paisajes_dificiles' aceptadas erróneamente:\n")
            for p in hard_failed:
                f.write(f"{p}\n")
            f.write("\n")

    if fallos_detectados:
        print(f"⚠️ Se registraron fallos. Revisa el archivo: {fallos_path}")
        raise RuntimeError("Validación fallida: tasa de error supera el umbral aceptable (30%)")

    return {
        "acceptance": {"errors": pos_errors, "total": pos_total},
        "rejection": {"errors": neg_errors, "total": neg_total},
        "hard_negatives": {"errors": hard_errors, "total": hard_total}
    }

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    dist_dir = cfg["dist_dir"]
    ensure_dir(dist_dir)

    print("Cargando modelo de embeddings:", cfg["embedding_model"])
    model = load_embedding_model(cfg["embedding_model"])
    extractor = EmbeddingExtractor(
        model=model,
        image_size=int(cfg["image_size"]),
        mean=cfg["normalize_mean"],
        std=cfg["normalize_std"]
    )

    # Solo usamos 'cadena' para construir el índice
    reference_paths_cadena = collect_images_from_class(cfg["cadena_dir"], "cadena")

    if cfg.get("max_reference_images"):
        reference_paths_cadena = reference_paths_cadena[:int(cfg["max_reference_images"])]

    if not reference_paths_cadena:
        raise RuntimeError("No se encontraron imágenes en la carpeta 'cadena'. Revisa rutas en config.yaml")

    print(f"Total de imágenes de referencia (cadena): {len(reference_paths_cadena)}")

    embeddings = []
    labels = []
    t0 = time.time()
    for path, cls in reference_paths_cadena:
        emb = extractor.extract_from_path(path)
        embeddings.append(emb.astype('float32'))
        labels.append(cls)
    t1 = time.time()

    embeddings = np.vstack(embeddings)
    print(f"Embeddings extraídos: {embeddings.shape} en {t1 - t0:.2f}s")

    idx = EmbeddingIndex(dim=int(cfg["embedding_dim"]))
    idx.add(embeddings, labels)

    artifacts = cfg["artifacts"]
    embeddings_path = os.path.join(dist_dir, artifacts["embeddings_npy"])
    labels_path = os.path.join(dist_dir, artifacts["labels_json"])
    faiss_path = os.path.join(dist_dir, artifacts["faiss_index"])
    manifest_path = os.path.join(dist_dir, artifacts["manifest_json"])
    report_path = os.path.join(dist_dir, artifacts["report_md"])

    np.save(embeddings_path, embeddings)
    idx.save(faiss_path, labels_path)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_model": cfg["embedding_model"],
        "embedding_dim": cfg["embedding_dim"],
        "image_size": cfg["image_size"],
        "reference_count": len(labels),
        "paths": {
            "embeddings_npy": embeddings_path,
            "labels_json": labels_path,
            "faiss_index": faiss_path
        },
        "thresholds": {
            "top_k": cfg["top_k"],
            "similarity_threshold": cfg["similarity_threshold"],
            "min_matches": cfg["min_matches"]
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    verifier = EmbeddingVerifier(
        faiss_index_path=faiss_path,
        labels_json_path=labels_path,
        embedding_dim=int(cfg["embedding_dim"]),
        top_k=int(cfg["top_k"]),
        similarity_threshold=float(cfg["similarity_threshold"]),
        min_matches=int(cfg["min_matches"])
    )

    # Validación extendida con negativos
    results = run_validation_tests(cfg, extractor, verifier)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Reporte de construcción de índice de embeddings\n\n")
        f.write(f"- Imágenes de referencia (cadena): {len(labels)}\n")
        f.write(f"- Dimensión de embedding: {cfg['embedding_dim']}\n")
        f.write(f"- Modelo: {cfg['embedding_model']}\n\n")
        f.write("## Resultados de validación extendida\n")
        for key, val in results.items():
            rate = val["errors"] / val["total"] if val["total"] else 0
            f.write(f"\n### {key.capitalize()}\n")
            f.write(f"- Total: {val['total']}\n")
            f.write(f"- Errores: {val['errors']}\n")
            f.write(f"- Tasa de error: {rate:.2%}\n")

    print("Listo. Artefactos generados en:", dist_dir)
    print("Embeddings:", embeddings_path)
    print("Labels:", labels_path)
    print("FAISS index:", faiss_path)
    print("Manifest:", manifest_path)
    print("Report:", report_path)
    
if __name__ == "__main__":
    main()