import os
import time
import json
from glob import glob
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from .core import (
    transform_tile, ecc_refine,
    compute_ssim, compute_mse, compute_lab_dist,
    compute_ncc, compute_edge_ncc, edge_strip
)
from .io import load_image, save_mosaic, save_params_json, save_params_csv

def compute_grad_ncc(a, b):
    ga = cv2.Sobel(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 1)
    gb = cv2.Sobel(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 1)
    return compute_ncc(ga, gb)


def list_tile_paths(folder):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths = []
    for e in exts:
        paths.extend(glob(os.path.join(folder, e)))
    return sorted(paths)

    
def _find_candidate(cand_list, tile_index):
    for c in cand_list:
        if c["tile_index"] == tile_index:
            return c
    return None

def _flip_str_to_mode(flip_str):
    if not flip_str:
        return None
    s = str(flip_str).lower()
    if s.startswith("h"):
        return 1
    if s.startswith("v"):
        return 0
    return None

def patch_position_prior(patch, row, total_rows):
    """
    Penalize patches that violate expected vertical structure.
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)

    # Expect brighter patches higher in image (sky)
    expected = 1.0 - (row / max(1, total_rows - 1))
    penalty = abs((mean_intensity / 255.0) - expected)

    return penalty


def _rotation_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)

def _translation_diff(a, b):
    ax, ay = a
    bx, by = b
    return abs(ax - bx) + abs(ay - by)

# ===================== NEW: safe translation subsampling =====================
def _subsample_translations(fracs, max_samples=3):
    """
    Subsample translation fractions symmetrically for coarse search.
    DOES NOT remove translations from final refinement.
    """
    fracs = sorted(set(fracs))
    if len(fracs) <= max_samples:
        return tuple(fracs)

    mid = len(fracs) // 2
    step = max(1, len(fracs) // max_samples)

    subsampled = fracs[mid::step]
    subsampled += fracs[:mid:step]
    subsampled = sorted(set(subsampled))

    return tuple(subsampled[:max_samples])


def run_pipeline(
    target_path, tiles_folder, out_image, out_params_json, out_params_csv,
    grid_n=10, grid_rows=None, grid_cols=None,
    coarse_width=200, top_k=5,
    rotations=(0, 90, 180, 270), flips=("none", "horizontal"),
    crop_fracs=(0.0,), zooms=(1.0,), translate_fracs=(-0.05, 0.0, 0.05),
    metric="hybrid", ecc_iter=100, ecc_eps=1e-5,
    alpha=0.0, resume=False,
    project_path="results/project/affair_project.json",
    progress_callback=None, max_coarse_ops=3_000_000
):
    t0 = time.time()

    if grid_rows is None or grid_cols is None:
        grid_rows = grid_cols = int(grid_n)

    os.makedirs(os.path.dirname(out_image) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_params_json) or ".", exist_ok=True)
    if project_path:
        os.makedirs(os.path.dirname(project_path), exist_ok=True)

    target = load_image(target_path)
    H, W = target.shape[:2]

    h_step = H // grid_rows
    w_step = W // grid_cols
    N = grid_rows * grid_cols

    # --- coarse scaling ---
    scale = float(coarse_width) / float(W) if coarse_width and W > coarse_width else 1.0
    small_w = max(2, int(round(W * scale)))
    small_h = max(2, int(round(H * scale)))

    small_target = cv2.resize(target, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_h_step = small_h // grid_rows
    small_w_step = small_w // grid_cols

    tile_paths = list_tile_paths(tiles_folder)
    if not tile_paths:
        raise FileNotFoundError("No tiles found")

    tiles_small = []
    for p in tile_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        t = cv2.resize(img, (small_w_step, small_h_step), interpolation=cv2.INTER_AREA)
        tiles_small.append((p, t))

    if not tiles_small:
        raise FileNotFoundError("No readable tiles")

    if progress_callback:
        progress_callback(1)

    candidates = None
    meta = {
        "target": os.path.abspath(target_path),
        "tiles": os.path.abspath(tiles_folder),
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
    }

    if resume and project_path and os.path.exists(project_path):
        try:
            with open(project_path, "r") as f:
                saved = json.load(f)
            if saved.get("_meta_") == meta:
                candidates = saved.get("candidates")
        except Exception:
            candidates = None

    if candidates is None:
        candidates = [[] for _ in range(N)]

    flips_list = tuple(flips) if flips else ("none",)

    total_ops = (
        N * len(tiles_small) * len(rotations)
        * len(flips_list) * len(crop_fracs)
        * len(zooms) * (len(translate_fracs) ** 2)
    )

    # 🔧 FIX: subsample translations ONLY for coarse search
    coarse_translate_fracs = translate_fracs
    if total_ops > max_coarse_ops:
        coarse_translate_fracs = _subsample_translations(
            translate_fracs, max_samples=3
        )

    # ===================== COARSE SEARCH =====================
    total_patches = grid_rows * grid_cols
    processed = 0
    
    for i in range(grid_rows):
        for j in range(grid_cols):
            pidx = i * grid_cols + j
            if resume and candidates[pidx]:
                continue

            y0, y1 = i * small_h_step, (i + 1) * small_h_step
            x0, x1 = j * small_w_step, (j + 1) * small_w_step
            patch = small_target[y0:y1, x0:x1]

            best = []

            for t_idx, (_, tile_small) in enumerate(tiles_small):
                for rot in rotations:
                    for flip in flips_list:
                        flip_mode = _flip_str_to_mode(flip)
                        for crop in crop_fracs:
                            for zoom in zooms:
                                for tx in coarse_translate_fracs:
                                    for ty in coarse_translate_fracs:

                                        warped = transform_tile(
                                            tile_small,
                                            patch.shape[:2],
                                            rotation=rot,
                                            flip=flip_mode,
                                            crop_frac=crop,
                                            zoom=zoom,
                                            translate=(tx, ty)
                                        )

                                        if metric == "ssim":
                                            score = compute_ssim(patch, warped)
                                        elif metric == "mse":
                                            score = -compute_mse(patch, warped)
                                        elif metric == "lab":
                                            score = -compute_lab_dist(patch, warped)
                                        elif metric == "ncc":
                                            score = compute_ncc(patch, warped)
                                            
                                        else:
                                            ssimv = compute_ssim(patch, warped)
                                            nccv = compute_ncc(patch, warped)
                                            gradv = compute_grad_ncc(patch, warped)
                                            labv = compute_lab_dist(patch, warped)

                                            score = (
                                                0.45 * ssimv +
                                                0.25 * nccv +
                                                0.30 * gradv -
                                                0.1 * (labv / 100)
                                                 )


                                        best.append(
                                            (score, t_idx, rot, flip_mode, crop, zoom, tx, ty)
                                        )

            best.sort(key=lambda x: x[0], reverse=True)
            # keep more candidates for global optimization
            K_KEEP = min(len(best), max(top_k, 50))
            best = best[:K_KEEP]

            candidates[pidx] = [
                {
                    "tile_index": int(t),
                    "rotation": int(r),
                    "flip_mode": f,
                    "crop_frac": float(c),
                    "zoom": float(z),
                    "translate": [float(tx), float(ty)],
                    "coarse_score": float(s)
                }
                for (s, t, r, f, c, z, tx, ty) in best
            ]

    if project_path:
        with open(project_path, "w") as f:
            json.dump({"_meta_": meta, "candidates": candidates}, f, indent=2)
            
    # ---- PROGRESS UPDATE ----
    processed += 1
    if progress_callback:
        progress_callback(int(100 * processed / total_patches))

    # ===================== ASSIGNMENT =====================
    ROT_LAMBDA = 0.05
    FLIP_LAMBDA = 0.2
    EDGE_LAMBDA = 1.0
    TRANS_LAMBDA = 1.2
    GLOBAL_ROT_LAMBDA = 2.0

    NUM_ASSIGN_ITERS = 4
    assignment = None

    for assign_iter in range(NUM_ASSIGN_ITERS):

        # ---- estimate dominant rotation (Phase 2) ----
        dominant_rotation = None
        if assignment:
            rots = []
            for r, cidx in assignment.items():
                cand = _find_candidate(candidates[r], cidx)
                if cand:
                    rots.append(cand["rotation"])
            if rots:
                dominant_rotation = max(set(rots), key=rots.count)

        cost = np.ones((N, len(tiles_small)), dtype=np.float32) * 10.0

        for i in range(N):
            pr = i // grid_cols
            pc = i % grid_cols

            for c in candidates[i]:
                idx = int(c["tile_index"])
                base_cost = 1.0 - c["coarse_score"]
                penalty = 0.0
                
                # ===================== PATCH POSITION PRIOR ===================
                PATCH_PRIOR_LAMBDA = 0.8

                y0 = pr * small_h_step
                y1 = (pr + 1) * small_h_step
                x0 = pc * small_w_step
                x1 = (pc + 1) * small_w_step

                patch = small_target[y0:y1, x0:x1]
                penalty += PATCH_PRIOR_LAMBDA * patch_position_prior(
                patch, pr, grid_rows
)


                # ---- global rotation penalty ----
                if dominant_rotation is not None:
                    penalty += GLOBAL_ROT_LAMBDA * (
                        _rotation_diff(c["rotation"], dominant_rotation) / 180.0
                    )

                # ---- LEFT neighbor ----
                if pc > 0 and assignment and (i - 1) in assignment:
                    left_idx = assignment[i - 1]
                    left = _find_candidate(candidates[i - 1], left_idx)
                    if left:
                        penalty += ROT_LAMBDA * (
                            _rotation_diff(c["rotation"], left["rotation"]) / 180.0
                        )
                        penalty += TRANS_LAMBDA * (
                            abs(c["translate"][0] - left["translate"][0]) +
                            abs(c["translate"][1] - left["translate"][1])
                        )
                        if c["flip_mode"] != left["flip_mode"]:
                            penalty += FLIP_LAMBDA
                        try:
                            ei = edge_strip(tiles_small[idx][1], "left")
                            ej = edge_strip(tiles_small[left_idx][1], "right")
                            penalty += EDGE_LAMBDA * (1.0 - compute_edge_ncc(ei, ej))
                        except Exception:
                            pass

                # ---- TOP neighbor ----
                if pr > 0 and assignment and (i - grid_cols) in assignment:
                    top_idx = assignment[i - grid_cols]
                    top = _find_candidate(candidates[i - grid_cols], top_idx)
                    if top:
                        penalty += ROT_LAMBDA * (
                            _rotation_diff(c["rotation"], top["rotation"]) / 180.0
                        )
                        penalty += TRANS_LAMBDA * (
                            abs(c["translate"][0] - top["translate"][0]) +
                            abs(c["translate"][1] - top["translate"][1])
                        )
                        if c["flip_mode"] != top["flip_mode"]:
                            penalty += FLIP_LAMBDA
                        try:
                            ei = edge_strip(tiles_small[idx][1], "top")
                            ej = edge_strip(tiles_small[top_idx][1], "bottom")
                            penalty += EDGE_LAMBDA * (1.0 - compute_edge_ncc(ei, ej))
                        except Exception:
                            pass

                cost[i, idx] = min(cost[i, idx], base_cost + penalty)

        cost += (np.random.rand(*cost.shape) - 0.5) * 1e-6
        rows, cols = linear_sum_assignment(cost)
        assignment = dict(zip(rows.tolist(), cols.tolist()))

    # ===================== ECC + MOSAIC =====================
    mosaic = np.zeros_like(target)
    records = []

    for r, c in assignment.items():
        cand = _find_candidate(candidates[r], c)

        pr = r // grid_cols
        pc = r % grid_cols
        y0, y1 = pr * h_step, (pr + 1) * h_step
        x0, x1 = pc * w_step, (pc + 1) * w_step
        dst = target[y0:y1, x0:x1]

        tile = cv2.imread(tile_paths[c])
        init = transform_tile(
            tile,
            dst.shape[:2],
            rotation=cand["rotation"] if cand else 0,
            flip=cand["flip_mode"] if cand else None,
            crop_frac=cand["crop_frac"] if cand else 0.0,
            zoom=cand["zoom"] if cand else 1.0,
            translate=tuple(cand["translate"]) if cand else (0, 0)
        )

        aligned, ok, _, _ = ecc_refine(init, dst, ecc_iter, ecc_eps)
        mosaic[y0:y1, x0:x1] = aligned

        records.append({
            "patch_row": pr,
            "patch_col": pc,
            "tile_index": c,
            "rotation": cand["rotation"] if cand else 0,
            "zoom": cand["zoom"] if cand else 1.0,
            "translate": cand["translate"] if cand else [0, 0],
            "ecc_success": bool(ok)
        })

    if alpha > 0:
        mosaic = cv2.addWeighted(mosaic, 1 - alpha, target, alpha, 0)

    save_mosaic(mosaic, out_image)
    save_params_json(records, out_params_json)
    save_params_csv(records, out_params_csv)

    # ---- FINAL progress update ----
    if progress_callback:
        progress_callback(100)

    return {
        "mosaic_path": out_image,
        "params_json": out_params_json,
        "params_csv": out_params_csv,
        "time_s": time.time() - t0
    }
