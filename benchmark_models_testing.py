# -*- coding: utf-8 -*-
"""
Benchmark comparativo entre YOLOv11-Small y YOLOv11-Nano
Evalúa ambos modelos sobre el dataset definido en data.yaml
y guarda métricas en CSV.
"""

import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import torch

# =========================
# Configuración
# =========================
DATASET = r"C:\Dansware_dev\modeloCadena\data.yaml"
MODELS_DIR = Path(r"C:\Dansware_dev\modeloCadena")
RUNS_DIR = Path(r"C:\Dansware_dev\modeloCadena\runs\benchmark_compare")

# Ajusta aquí las rutas a tus pesos entrenados
YOLO11S = MODELS_DIR / "best.pt"  # tu modelo YOLOv11-Small entrenado
YOLO11N = MODELS_DIR / "runs" / "detect" / "retrain_736_nano" / "weights" / "best.pt"  # tu YOLOv11-Nano entrenado

IMGSZ = 736
TASK = "detect"

def has_cuda():
    return torch.cuda.is_available()

def evaluate_model(label, model_path, device="cpu", save_dir=None):
    """Evalúa un modelo YOLO y devuelve métricas principales."""
    print(f"\n🔎 Evaluando {label} en {device} ...")
    model = YOLO(str(model_path), task=TASK)
    results = model.val(
        data=DATASET,
        imgsz=IMGSZ,
        device=device,
        project=str(save_dir.parent) if save_dir else None,
        name=save_dir.name if save_dir else None,
        verbose=False
    )
    metrics = results.results_dict
    speed = results.speed
    return {
        "Modelo": label,
        "Ruta": str(model_path),
        "Dispositivo": device,
        "mAP50": round(metrics.get("metrics/mAP50(B)", 0.0), 3),
        "mAP50-95": round(metrics.get("metrics/mAP50-95(B)", 0.0), 3),
        "Precisión": round(metrics.get("metrics/precision(B)", 0.0), 3),
        "Recall": round(metrics.get("metrics/recall(B)", 0.0), 3),
        "Latencia (ms/img)": round(speed.get("inference", 0.0), 3),
        "Estado": "OK"
    }

def main():
    resultados = []
    device = "0" if has_cuda() else "cpu"

    if YOLO11S.exists():
        resultados.append(evaluate_model("YOLOv11-Small", YOLO11S, device=device, save_dir=RUNS_DIR / "yolo11s"))
    else:
        print("⚠️ No se encontró YOLOv11-Small")

    if YOLO11N.exists():
        resultados.append(evaluate_model("YOLOv11-Nano", YOLO11N, device=device, save_dir=RUNS_DIR / "yolo11n"))
    else:
        print("⚠️ No se encontró YOLOv11-Nano")

    # Guardar resultados
    df = pd.DataFrame(resultados)
    print("\n📊 Resultados comparativos:\n")
    print(df.to_string(index=False))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RUNS_DIR / "benchmark_yolo11s_vs_yolo11n.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n💾 Resultados guardados en: {out_csv}")

if __name__ == "__main__":
    main()