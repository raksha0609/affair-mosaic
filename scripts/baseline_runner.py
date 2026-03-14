import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import time
from glob import glob

import cv2
import numpy as np

from mosaic.core import (
    compute_ssim, compute_ncc, compute_mse, compute_lab_dist,
    ecc_refine, transform_tile, to_gray
)
from mosaic.io import load_image, save_mosaic, save_params_json, save_params_csv


# compute_grad_ncc lives in pipeline.py, not core.py — define it locally
def compute_grad_ncc(a, b):
    ga = cv2.Sobel(to_gray(a).astype("float32"), cv2.CV_32F, 1, 1)
    gb = cv2.Sobel(to_gray(b).astype("float32"), cv2.CV_32F, 1, 1)
    return compute_ncc(ga, gb)


def list_tiles(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    paths = []
    for e in exts:
        paths.extend(glob(os.path.join(folder, e)))
    return sorted(paths)


def compute_hybrid_score(patch, tile):
    ssimv = compute_ssim(patch, tile)
    nccv  = compute_ncc(patch, tile)
    gradv = compute_grad_ncc(patch, tile)
    labv  = compute_lab_dist(patch, tile)
    return 0.45 * ssimv + 0.25 * nccv + 0.30 * gradv - 0.1 * (labv / 100.0)


# ----------------------------------------------------------------------------
# Random baseline
# ----------------------------------------------------------------------------

def run_random(target, tile_paths, grid_rows, grid_cols,
               h_step, w_step, ecc_iter, ecc_eps):
    N = grid_rows * grid_cols
    indices = list(range(len(tile_paths)))
    assignment_list = (indices * (N // len(indices) + 1))[:N]
    random.shuffle(assignment_list)

    mosaic  = np.zeros_like(target)
    records = []

    for idx, tile_idx in enumerate(assignment_list):
        pr = idx // grid_cols
        pc = idx % grid_cols
        y0, y1 = pr * h_step, (pr + 1) * h_step
        x0, x1 = pc * w_step, (pc + 1) * w_step
        dst = target[y0:y1, x0:x1]

        tile = cv2.imread(tile_paths[tile_idx])
        init = transform_tile(tile, dst.shape[:2])
        aligned, ok, _, _ = ecc_refine(init, dst, ecc_iter, ecc_eps)
        mosaic[y0:y1, x0:x1] = aligned

        records.append({
            "patch_row":   pr,
            "patch_col":   pc,
            "tile_index":  tile_idx,
            "rotation":    0,
            "zoom":        1.0,
            "translate":   [0.0, 0.0],
            "ecc_success": bool(ok)
        })

        pct = int(100 * (idx + 1) / N)
        print(f"\r  placing tiles... {pct}%", end="", flush=True)

    print()
    return mosaic, records


# ----------------------------------------------------------------------------
# Greedy baseline
# ----------------------------------------------------------------------------

def run_greedy(target, tile_paths, grid_rows, grid_cols,
               h_step, w_step, coarse_width,
               metric, ecc_iter, ecc_eps):
    H, W = target.shape[:2]
    scale        = float(coarse_width) / float(W) if coarse_width and W > coarse_width else 1.0
    small_w      = max(2, int(round(W * scale)))
    small_h      = max(2, int(round(H * scale)))
    small_target = cv2.resize(target, (small_w, small_h))

    small_h_step = small_h // grid_rows
    small_w_step = small_w // grid_cols

    tiles_small = []
    for p in tile_paths:
        img = cv2.imread(p)
        if img is not None:
            t = cv2.resize(img, (small_w_step, small_h_step))
            tiles_small.append((p, t))

    if not tiles_small:
        raise FileNotFoundError("No readable tiles found.")

    mosaic  = np.zeros_like(target)
    records = []
    N       = grid_rows * grid_cols

    for pidx in range(N):
        pr = pidx // grid_cols
        pc = pidx % grid_cols

        y0s = pr * small_h_step
        y1s = (pr + 1) * small_h_step
        x0s = pc * small_w_step
        x1s = (pc + 1) * small_w_step
        patch_small = small_target[y0s:y1s, x0s:x1s]

        best_score = -1e9
        best_tidx  = 0
        for tidx, (_, t_small) in enumerate(tiles_small):
            if metric == "ssim":
                s = compute_ssim(patch_small, t_small)
            elif metric == "ncc":
                s = compute_ncc(patch_small, t_small)
            elif metric == "mse":
                s = -compute_mse(patch_small, t_small)
            else:
                s = compute_hybrid_score(patch_small, t_small)

            if s > best_score:
                best_score = s
                best_tidx  = tidx

        y0, y1 = pr * h_step, (pr + 1) * h_step
        x0, x1 = pc * w_step, (pc + 1) * w_step
        dst    = target[y0:y1, x0:x1]
        tile   = cv2.imread(tiles_small[best_tidx][0])
        init   = transform_tile(tile, dst.shape[:2])
        aligned, ok, _, _ = ecc_refine(init, dst, ecc_iter, ecc_eps)
        mosaic[y0:y1, x0:x1] = aligned

        records.append({
            "patch_row":   pr,
            "patch_col":   pc,
            "tile_index":  best_tidx,
            "rotation":    0,
            "zoom":        1.0,
            "translate":   [0.0, 0.0],
            "ecc_success": bool(ok)
        })

        pct = int(100 * (pidx + 1) / N)
        print(f"\r  scoring patches... {pct}%", end="", flush=True)

    print()
    return mosaic, records


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

def print_summary(records, mode, elapsed):
    total   = len(records)
    success = sum(1 for r in records if r["ecc_success"])
    print(f"\n{'='*55}")
    print(f"  Baseline  : {mode.upper()}")
    print(f"  Patches   : {total}")
    print(f"  ECC OK    : {success}  ({100 * success / max(1, total):.1f}%)")
    print(f"  ECC fail  : {total - success}  ({100 * (total - success) / max(1, total):.1f}%)")
    print(f"  Runtime   : {elapsed:.1f} s")
    print(f"{'='*55}")
    print("  Next: run phantom_eval.py --res <params> --mosaic <mosaic>")
    print(f"{'='*55}\n")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comparison baselines for AFFAIR (random or greedy)."
    )
    parser.add_argument("--target",     required=True)
    parser.add_argument("--tiles",      required=True)
    parser.add_argument("--mode",       required=True, choices=["random", "greedy"])
    parser.add_argument("--grid",       type=int, default=6)
    parser.add_argument("--grid-rows",  type=int, default=None)
    parser.add_argument("--grid-cols",  type=int, default=None)
    parser.add_argument("--coarse",     type=int, default=200)
    parser.add_argument("--metric",     default="hybrid",
                        choices=["hybrid", "ssim", "ncc", "mse"])
    parser.add_argument("--ecc-iter",   type=int, default=100)
    parser.add_argument("--ecc-eps",    type=float, default=1e-5)
    parser.add_argument("--out-mosaic", default="results/baselines/baseline_mosaic.jpg")
    parser.add_argument("--out-params", default="results/baselines/baseline_params.json")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    grid_rows  = args.grid_rows or args.grid
    grid_cols  = args.grid_cols or args.grid
    target     = load_image(args.target)
    H, W       = target.shape[:2]
    h_step     = H // grid_rows
    w_step     = W // grid_cols
    tile_paths = list_tiles(args.tiles)

    if not tile_paths:
        raise FileNotFoundError(f"No tiles found in: {args.tiles}")

    os.makedirs(os.path.dirname(args.out_mosaic) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_params)  or ".", exist_ok=True)

    print(f"\n[AFFAIR Baseline]  mode={args.mode}  grid={grid_rows}x{grid_cols}")
    print(f"  target : {args.target}")
    print(f"  tiles  : {len(tile_paths)} found in {args.tiles}\n")

    t0 = time.time()

    if args.mode == "random":
        mosaic, records = run_random(
            target, tile_paths, grid_rows, grid_cols,
            h_step, w_step, args.ecc_iter, args.ecc_eps
        )
    else:
        mosaic, records = run_greedy(
            target, tile_paths, grid_rows, grid_cols,
            h_step, w_step, args.coarse,
            args.metric, args.ecc_iter, args.ecc_eps
        )

    elapsed = time.time() - t0
    save_mosaic(mosaic, args.out_mosaic)
    save_params_json(records, args.out_params)

    print(f"  Mosaic saved : {args.out_mosaic}")
    print(f"  Params saved : {args.out_params}")
    print_summary(records, args.mode, elapsed)


if __name__ == "__main__":
    main()