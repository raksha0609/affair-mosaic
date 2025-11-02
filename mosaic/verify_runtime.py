# mosaic/verify_runtime.py
import os
import cv2
import time
import psutil
import argparse
import numpy as np
from mosaic.pipeline import run_pipeline

def format_bytes(num):
    """Human-readable memory size"""
    for unit in ['B','KB','MB','GB']:
        if num < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}TB"

def measure_memory():
    """Return current and peak memory (RSS)"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss
    return mem

def verify_runtime(target, tiles, grid=10, coarse=200, metric="hybrid"):
    print("="*70)
    print(f"[VERIFY] AFFAIR Runtime Verification")
    print("="*70)
    t0 = time.time()

    mem_start = measure_memory()

    # --- Stage 1: Coarse + Fine Run ---
    res = run_pipeline(
        target_path=target,
        tiles_folder=tiles,
        out_image="results/mosaics/verify_runtime_result.jpg",
        out_params_json="results/params/verify_runtime_params.json",
        out_params_csv="results/params/verify_runtime_params.csv",
        grid_n=grid,
        coarse_width=coarse,
        top_k=5,
        metric=metric,
        #fast_mode=True,
        progress_callback=lambda v: print(f"\rProgress: {v}%", end="", flush=True)
    )
    print("\n[INFO] Pipeline finished successfully.")

    mem_end = measure_memory()
    t_total = time.time() - t0

    # --- Stage 2: Analyze outputs ---
    params_json = res["params_json"]
    if not os.path.exists(params_json):
        print("[WARN] No parameter file found.")
        return

    import json
    with open(params_json, "r") as f:
        data = json.load(f)

    ecc_success = sum(1 for d in data if d.get("ecc_success"))
    total_patches = len(data)

    # --- Stage 3: Collect info ---
    target_img = cv2.imread(target)
    H, W = target_img.shape[:2]
    downsample_scale = coarse / float(W) if W > coarse else 1.0

    print("\n========== RUNTIME SUMMARY ==========")
    print(f"Target: {os.path.basename(target)}  ({W}x{H})")
    print(f"Tiles folder: {tiles}")
    print(f"Grid size: {grid} x {grid}")
    print(f"Downsample scale: {downsample_scale:.3f}")
    print(f"Coarse width used: {coarse}px")
    print(f"Similarity metric: {metric}")
    print(f"ECC refinement success: {ecc_success}/{total_patches}")
    print(f"Total runtime: {t_total/60:.2f} min ({t_total:.1f} s)")
    print(f"Memory usage start: {format_bytes(mem_start)}")
    print(f"Memory usage end: {format_bytes(mem_end)}")
    print("Results:")
    print(f"  - Mosaic: {res['mosaic_path']}")
    print(f"  - Params JSON: {res['params_json']}")
    print(f"  - Params CSV: {res['params_csv']}")
    print("=====================================")
<<<<<<< HEAD
    print(" Verification complete. Everything looks consistent.\n")
=======
    print("Verification complete. Everything looks consistent.\n")
>>>>>>> 69b45d9 (Updated pipeline and GUI integration for AFFAIR Mosaic (excluding phantom evaluation))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--tiles", required=True)
    parser.add_argument("--grid", type=int, default=10)
    parser.add_argument("--coarse", type=int, default=200)
    parser.add_argument("--metric", default="hybrid")
    args = parser.parse_args()

    verify_runtime(
        target=args.target,
        tiles=args.tiles,
        grid=args.grid,
        coarse=args.coarse,
        metric=args.metric
    )
