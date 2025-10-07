# mosaic/pipeline.py
import os
import time
import json
from glob import glob
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from .core import (
    center_crop_to_multiple, transform_tile, ecc_refine,
    compute_ssim, compute_mse, compute_lab_dist, compute_ncc
)
from .io import load_image, save_mosaic, save_params_json, save_params_csv

def list_tile_paths(folder):
    return sorted(glob(os.path.join(folder, "*.jpg")) + glob(os.path.join(folder, "*.png")))

def run_pipeline(target_path, tiles_folder, out_image, out_params_json, out_params_csv,
                 grid_n=10, coarse_width=200, top_k=5,
                 rotations=(0,90,180,270), flips=(False, True),
                 crop_fracs=(0.0, 0.2), zooms=(1.0,), translate_fracs=(0.0,),
                 metric="hybrid", ecc_iter=100, ecc_eps=1e-5,
                 alpha=0.0, resume=False, checkpoint_path="results/params/coarse_candidates.json",
                 progress_callback=None, max_coarse_ops=5_000_000):
    """
    Steps:
     - center-crop target to multiple of grid
     - create small (coarse) target for fast search (coarse_width)
     - for each patch compute top_k candidates (try transforms), store only parameters + score
     - save coarse candidates checkpoint (JSON)
     - build cost matrix (N x M) with big cost default, fill with candidate scores
     - Hungarian assignment
     - refine assigned pairs using ECC on full-resolution tiles (load tiles lazily)
     - save mosaic + params
    """
    t0 = time.time()
    target = load_image(target_path)
    target, h_step, w_step = center_crop_to_multiple(target, grid_n)
    H, W = target.shape[:2]
    N = grid_n * grid_n

    # coarse downsample target
    if coarse_width and W > coarse_width:
        scale = coarse_width / float(W)
    else:
        scale = 1.0
    small_target = cv2.resize(target, (max(2,int(round(W*scale))), max(2,int(round(H*scale)))))
    small_h, small_w = small_target.shape[:2]
    small_h_step = small_h // grid_n
    small_w_step = small_w // grid_n

    # collect tile paths
    tile_paths = list_tile_paths(tiles_folder)
    if len(tile_paths) == 0:
        raise FileNotFoundError("No tiles found")

    # Precompute coarse-size versions of tiles to accelerate transforms
    tiles_small = []
    for p in tile_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        # NOTE: we keep only the small version for coarse stage here
        small = cv2.resize(img, (small_w_step, small_h_step), interpolation=cv2.INTER_AREA)
        tiles_small.append((p, small))
    M = len(tiles_small)
    if M == 0:
        raise FileNotFoundError("No readable tiles for coarse stage")

    if progress_callback: progress_callback(1)

    # If resume and checkpoint exists, load it
    if resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            candidates = json.load(f)
        # ensure length
        if not isinstance(candidates, list) or len(candidates) != N:
            # ignore checkpoint if inconsistent
            candidates = [[] for _ in range(N)]
    else:
        candidates = [[] for _ in range(N)]

    # compute total coarse ops to guard against explosion
    total_ops = N * M * max(1, len(rotations)) * max(1, len(flips)) * max(1, len(crop_fracs)) * max(1, len(zooms)) * max(1, len(translate_fracs))**2
    if total_ops > max_coarse_ops:
        # too many combinations; lower transforms automatically: warn but continue with limited set
        # We'll reduce translate_fracs to single 0.0 if explosion risk
        translate_fracs = tuple(list(translate_fracs)[:1])
    done = 0

    # Coarse search: keep only param dicts (no images)
    for i in range(grid_n):
        for j in range(grid_n):
            pidx = i * grid_n + j
            # skip if already have candidates (resume)
            if resume and candidates[pidx]:
                # still update progress estimations (rough)
                if progress_callback:
                    progress_callback(1 + int(39 * (pidx+1) / N))
                continue

            y0, y1 = i*small_h_step, (i+1)*small_h_step
            x0, x1 = j*small_w_step, (j+1)*small_w_step
            patch_small = small_target[y0:y1, x0:x1]

            # keep top_k list of (score, params...)
            best = []

            for t_idx, (tpath, t_small) in enumerate(tiles_small):
                for rot in rotations:
                    for flip in flips:
                        for crop in crop_fracs:
                            for zoom in zooms:
                                # translate combinations: use tx,ty from translate_fracs
                                for tx in translate_fracs:
                                    for ty in translate_fracs:
                                        # transform small tile
                                        trans = transform_tile(t_small, patch_small.shape[:2],
                                                               rotation=rot, flip=flip,
                                                               crop_frac=crop, zoom=zoom,
                                                               translate=(tx, ty))
                                        # compute requested metric
                                        ssimv = compute_ssim(patch_small, trans)
                                        msev = compute_mse(patch_small, trans)
                                        labv = compute_lab_dist(patch_small, trans)
                                        nccv = compute_ncc(patch_small, trans)
                                        if metric == "ssim":
                                            score = ssimv
                                        elif metric == "mse":
                                            score = -msev
                                        elif metric == "lab":
                                            score = -labv
                                        elif metric == "ncc":
                                            score = nccv
                                        elif metric == "hybrid":
                                            # hybrid example: favor ssim, penalize lab distance
                                            score = ssimv - 0.25 * (labv / 100.0)
                                        else:
                                            score = ssimv

                                        # push into best list
                                        if len(best) < top_k:
                                            best.append((score, t_idx, rot, flip, crop, zoom, tx, ty))
                                            best.sort(reverse=True, key=lambda x: x[0])
                                        else:
                                            if score > best[-1][0]:
                                                best[-1] = (score, t_idx, rot, flip, crop, zoom, tx, ty)
                                                best.sort(reverse=True, key=lambda x: x[0])

                                        done += 1
                # update coarse progress occasionally
                if progress_callback and done % 2000 == 0:
                    progress_callback(1 + int(38 * done / max(1, total_ops)))

            # store params-only candidate dicts for this patch
            candidates[pidx] = []
            for (s, t_idx, rot, flip, crop, zoom, tx, ty) in best:
                candidates[pidx].append({
                    "tile_index": int(t_idx),
                    "rotation": int(rot),
                    "flip": bool(flip),
                    "crop_frac": float(crop),
                    "zoom": float(zoom),
                    "translate": [float(tx), float(ty)],
                    "coarse_score": float(s)
                })

            # small progress jump
            if progress_callback:
                progress_callback(1 + int(38 * (pidx+1) / N))

    # save coarse checkpoint
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(candidates, f, indent=2)

    if progress_callback:
        progress_callback(40)

    # Build cost matrix: default big cost, fill candidate entries
    big_cost = 10.0
    cost = np.ones((N, M), dtype=float) * big_cost
    for pidx in range(N):
        for cand in candidates[pidx]:
            t_idx = int(cand["tile_index"])
            cost[pidx, t_idx] = min(cost[pidx, t_idx], 1.0 - float(cand.get("coarse_score", 0.0)))

    if progress_callback:
        progress_callback(45)

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    if progress_callback:
        progress_callback(55)

    # Refinement: load full-resolution tiles lazily only for assigned ones
    mosaic = np.zeros_like(target)
    records = []
    for k, r in enumerate(row_ind):
        c = int(col_ind[k])
        # find best candidate for this (patch r and tile c)
        cand_list = [cnd for cnd in candidates[r] if int(cnd["tile_index"]) == c]
        cand = None
        if cand_list:
            # choose highest coarse_score
            cand = max(cand_list, key=lambda x: x.get("coarse_score", -999.0))

        # compute dst_patch coordinates at full resolution
        pr = r // grid_n
        pc = r % grid_n
        y0, y1 = pr * h_step, (pr+1) * h_step
        x0, x1 = pc * w_step, (pc+1) * w_step
        dst_patch = target[y0:y1, x0:x1]

        # load full tile image and apply candidate transforms
        tile_full_path = tile_paths[c]
        tile_full = cv2.imread(tile_full_path)
        if tile_full is None:
            init = np.zeros_like(dst_patch)
            ok = False
            warp = None
            final_score = -1.0
        else:
            if cand is None:
                init = cv2.resize(tile_full, (dst_patch.shape[1], dst_patch.shape[0]))
            else:
                init = transform_tile(tile_full, dst_patch.shape[:2],
                                      rotation=cand["rotation"], flip=cand["flip"],
                                      crop_frac=cand["crop_frac"], zoom=cand["zoom"],
                                      translate=tuple(cand["translate"]))
            aligned, ok, warp, final_score = ecc_refine(init, dst_patch, max_iter=ecc_iter, eps=ecc_eps)
            init = aligned if ok else init

        mosaic[y0:y1, x0:x1] = init

        records.append({
            "patch_index": int(r),
            "patch_row": int(pr),
            "patch_col": int(pc),
            "tile_index": int(c),
            "tile_path": os.path.basename(tile_full_path),
            "rotation": int(cand["rotation"]) if cand else 0,
            "flip": bool(cand["flip"]) if cand else False,
            "crop_frac": float(cand["crop_frac"]) if cand else 0.0,
            "zoom": float(cand["zoom"]) if cand else 1.0,
            "translate": list(cand["translate"]) if cand else [0.0, 0.0],
            "coarse_score": float(cand["coarse_score"]) if cand else 0.0,
            "ecc_success": bool(ok),
            "ecc_score": float(final_score),
            "warp": warp.tolist() if hasattr(warp, "tolist") else (warp.tolist() if warp is not None else None)
        })

        if progress_callback:
            progress_callback(55 + int(40 * (k+1) / max(1, len(row_ind))))

    # optional blend with original target
    if alpha and alpha > 0:
        mosaic = cv2.addWeighted(mosaic, 1.0 - alpha, target, alpha, 0.0)

    # save
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
