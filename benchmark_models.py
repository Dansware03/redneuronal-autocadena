# -*- coding: utf-8 -*-
"""
Benchmark de modelos YOLO exportados.
Primero valida el modelo base best.pt y guarda resultados en runs/testing.
Luego evalúa los demás formatos exportados.
"""

import os
import torch
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# =========================
# Configuración
# =========================
DATASET = r"C:\Dansware_dev\modeloCadena\data.yaml"
MODELS_DIR = Path(r"C:\Dansware_dev\AutoCadena\models")
RUNS_DIR = Path(r"C:\Dansware_dev\modeloCadena\runs\testing")

# Modelos exportados
BEST_PT = MODELS_DIR / "best.pt"
ONNX_NVIDIA = MODELS_DIR / "best_nvidia_cuda.onnx"
ONNX_AMD = MODELS_DIR / "best_amd_directml.onnx"
ONNX_CPU = MODELS_DIR / "best_cpu.onnx"
OV_DIR = MODELS_DIR / "best_openvino_model"

IMGSZ = 736
TASK = "detect"

# =========================
# Utilidades
# =========================
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
        "Formato": label,
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

    # 1) Validar primero el modelo base best.pt
    if BEST_PT.exists():
        resultados.append(
            evaluate_model("Best (.pt)", BEST_PT,
                           device="0" if has_cuda() else "cpu",
                           save_dir=RUNS_DIR)
        )
    else:
        print(f"⚠️ No se encontró {BEST_PT}")

    # 2) Evaluar ONNX NVIDIA
    if ONNX_NVIDIA.exists():
        resultados.append(evaluate_model("ONNX (NVIDIA CUDA)", ONNX_NVIDIA,
                                         device="0" if has_cuda() else "cpu"))

    # 3) Evaluar ONNX AMD (DirectML) - en tu laptop se probará en CPU
    if ONNX_AMD.exists():
        resultados.append(evaluate_model("ONNX (AMD DirectML)", ONNX_AMD,
                                         device="cpu"))

    # 4) Evaluar ONNX CPU genérico
    if ONNX_CPU.exists():
        resultados.append(evaluate_model("ONNX (CPU)", ONNX_CPU,
                                         device="cpu"))

    # 5) Evaluar OpenVINO (Intel CPU)
    if OV_DIR.exists():
        resultados.append(evaluate_model("OpenVINO (Intel CPU)", OV_DIR,
                                         device="cpu"))

    # 6) Guardar resultados en CSV dentro de runs/testing
    df = pd.DataFrame(resultados)
    print("\n📊 Resultados comparativos:\n")
    print(df.to_string(index=False))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RUNS_DIR / "benchmark_results.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n💾 Resultados guardados en: {out_csv}")

if __name__ == "__main__":
    main()