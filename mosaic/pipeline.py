import os
import time
import json
from glob import glob
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from .core import (
    transform_tile, ecc_refine,
    compute_ssim, compute_mse, compute_lab_dist, compute_ncc
)
from .io import load_image, save_mosaic, save_params_json, save_params_csv


def list_tile_paths(folder):
    return sorted(glob(os.path.join(folder, "*.jpg")) + glob(os.path.join(folder, "*.png")))


def run_pipeline(target_path, tiles_folder, out_image, out_params_json, out_params_csv,
                 grid_n=10, coarse_width=200, top_k=5,
                 rotations=(0, 90, 180, 270), flips=("none", "horizontal"),
                 crop_fracs=(0.0, 0.2), zooms=(1.0,), translate_fracs=(0.0,),
                 metric="hybrid", ecc_iter=100, ecc_eps=1e-5,
                 alpha=0.0, resume=False,
                 project_path="results/project/affair_project.json",
                 progress_callback=None, max_coarse_ops=3_000_000):
    """
    AFFAIR stable pipeline (fast & tested version)
      - Coarse rigid transform search
      - Global assignment (Hungarian)
      - ECC rigid fine alignment
      - Hybrid similarity support
    """
    t0 = time.time()
    os.makedirs(os.path.dirname(project_path), exist_ok=True)

    # Load target
    target = load_image(target_path)
    H, W = target.shape[:2]
    h_step, w_step = H // grid_n, W // grid_n
    N = grid_n * grid_n

    # Downsample target for coarse search
    scale = coarse_width / float(W) if coarse_width and W > coarse_width else 1.0
    small_target = cv2.resize(target, (int(W * scale), int(H * scale)))
    small_h, small_w = small_target.shape[:2]
    small_h_step, small_w_step = small_h // grid_n, small_w // grid_n

    # Load tiles
    tile_paths = list_tile_paths(tiles_folder)
    if not tile_paths:
        raise FileNotFoundError("No tiles found in folder")

    tiles_small = []
    for p in tile_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        small = cv2.resize(img, (small_w_step, small_h_step), interpolation=cv2.INTER_AREA)
        tiles_small.append((p, small))
    if not tiles_small:
        raise FileNotFoundError("No readable tiles")

    if progress_callback:
        progress_callback(1)

    # Resume check
    candidates = None
    if resume and os.path.exists(project_path):
        try:
            with open(project_path, "r") as f:
                saved = json.load(f)
            header = saved.get("_meta_", {})
            if header.get("target") == os.path.abspath(target_path):
                candidates = saved.get("candidates", [])
                if progress_callback:
                    progress_callback(5)
        except Exception:
            print("[WARN] Resume failed — starting fresh")

    if candidates is None:
        candidates = [[] for _ in range(N)]

    # Coarse search
    total_ops = N * len(tiles_small) * len(rotations) * len(flips) * len(crop_fracs) * len(zooms)
    done = 0
    for i in range(grid_n):
        for j in range(grid_n):
            pidx = i * grid_n + j
            if resume and candidates[pidx]:
                continue

            y0, y1 = i * small_h_step, (i + 1) * small_h_step
            x0, x1 = j * small_w_step, (j + 1) * small_w_step
            patch_small = small_target[y0:y1, x0:x1]
            best = []

            for t_idx, (tpath, t_small) in enumerate(tiles_small):
                for rot in rotations:
                    for flip in flips:
                        flip_mode = 1 if flip.lower().startswith("h") else None
                        for crop in crop_fracs:
                            for zoom in zooms:
                                trans = transform_tile(
                                    t_small, patch_small.shape[:2],
                                    rotation=rot, flip=flip_mode,
                                    crop_frac=crop, zoom=zoom
                                )

                                # Compute metric
                                if metric == "ssim":
                                    score = compute_ssim(patch_small, trans)
                                elif metric == "mse":
                                    score = -compute_mse(patch_small, trans)
                                elif metric == "lab":
                                    score = -compute_lab_dist(patch_small, trans)
                                elif metric == "ncc":
                                    score = 0.7 * compute_ncc(patch_small, trans) + 0.3 * compute_ssim(patch_small, trans)
                                elif metric == "hybrid":
                                    ssimv = compute_ssim(patch_small, trans)
                                    labv = compute_lab_dist(patch_small, trans)
                                    score = ssimv - 0.25 * (labv / 100.0)
                                else:
                                    score = compute_ssim(patch_small, trans)

                                if len(best) < top_k:
                                    best.append((score, t_idx, rot, flip_mode, crop, zoom))
                                    best.sort(reverse=True, key=lambda x: x[0])
                                elif score > best[-1][0]:
                                    best[-1] = (score, t_idx, rot, flip_mode, crop, zoom)
                                    best.sort(reverse=True, key=lambda x: x[0])

                                done += 1
                if progress_callback and done % 2000 == 0:
                    progress_callback(1 + int(38 * done / max(1, total_ops)))

            candidates[pidx] = [
                {
                    "tile_index": int(t_idx),
                    "rotation": int(rot),
                    "flip_mode": flip_mode,
                    "crop_frac": float(crop),
                    "zoom": float(zoom),
                    "coarse_score": float(s)
                }
                for (s, t_idx, rot, flip_mode, crop, zoom) in best
            ]

    with open(project_path, "w") as f:
        json.dump({"_meta_": {"target": os.path.abspath(target_path)}, "candidates": candidates}, f, indent=2)
    if progress_callback:
        progress_callback(50)

    # Global optimization
    cost = np.ones((N, len(tiles_small)), dtype=float) * 10.0
    for pidx in range(N):
        for cand in candidates[pidx]:
            t_idx = int(cand["tile_index"])
            cost[pidx, t_idx] = min(cost[pidx, t_idx], 1.0 - float(cand["coarse_score"]))
    rows, cols = linear_sum_assignment(cost)
    if progress_callback:
        progress_callback(60)

    # Assemble mosaic
    mosaic = np.zeros_like(target)
    records = []
    for k, r in enumerate(rows):
        c = int(cols[k])
        cand_list = [x for x in candidates[r] if int(x["tile_index"]) == c]
        cand = max(cand_list, key=lambda x: x["coarse_score"]) if cand_list else None

        pr, pc = divmod(r, grid_n)
        y0, y1 = pr * h_step, (pr + 1) * h_step
        x0, x1 = pc * w_step, (pc + 1) * w_step
        dst_patch = target[y0:y1, x0:x1]

        tile_full_path = tile_paths[c]
        tile_full = cv2.imread(tile_full_path)
        if tile_full is None:
            aligned = np.zeros_like(dst_patch)
            ok, warp, final_score = False, None, -1.0
        else:
            init = transform_tile(
                tile_full, dst_patch.shape[:2],
                rotation=cand["rotation"], flip=cand["flip_mode"],
                crop_frac=cand["crop_frac"], zoom=cand["zoom"]
            ) if cand else cv2.resize(tile_full, (dst_patch.shape[1], dst_patch.shape[0]))
            aligned, ok, warp, final_score = ecc_refine(init, dst_patch, max_iter=ecc_iter, eps=ecc_eps)
            if not ok:
                aligned = init

        mosaic[y0:y1, x0:x1] = aligned
        records.append({
            "patch_index": int(r),
            "patch_row": int(pr),
            "patch_col": int(pc),
            "tile_index": int(c),
            "tile_path": os.path.basename(tile_full_path),
            "rotation": int(cand["rotation"]) if cand else 0,
            "flip_mode": cand["flip_mode"] if cand else None,
            "crop_frac": float(cand["crop_frac"]) if cand else 0.0,
            "zoom": float(cand["zoom"]) if cand else 1.0,
            "coarse_score": float(cand["coarse_score"]) if cand else 0.0,
            "ecc_success": bool(ok),
            "ecc_score": float(final_score)
        })

        if progress_callback:
            progress_callback(60 + int(35 * (k + 1) / len(rows)))

    if alpha > 0:
        mosaic = cv2.addWeighted(mosaic, 1.0 - alpha, target, alpha, 0.0)

    save_mosaic(mosaic, out_image)
    save_params_json(records, out_params_json)
    save_params_csv(records, out_params_csv)

    if progress_callback:
        progress_callback(100)

    return {
        "mosaic_path": out_image,
        "params_json": out_params_json,
        "params_csv": out_params_csv,
        "time_s": time.time() - t0
    }
