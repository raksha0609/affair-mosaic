# mosaic/pipeline.py
import os, time, cv2, json
import numpy as np
from glob import glob
from scipy.optimize import linear_sum_assignment
from .core import (center_crop_to_multiple, transform_tile, ecc_refine, compute_ssim)
from .io import load_image, save_mosaic, save_params_json, save_params_csv

def list_tile_paths(folder):
    return sorted(glob(os.path.join(folder, "*.jpg")) + glob(os.path.join(folder, "*.png")))

def compute_lab_dist(a, b):
    labA = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype("float32")
    labB = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype("float32")
    return np.mean((labA - labB) ** 2) ** 0.5

def run_pipeline(target_path, tiles_folder, out_image, out_params_json, out_params_csv,
                 grid_n=10, coarse_width=200, top_k=5,
                 rotations=(0,90,180,270), flips=(False, True),
                 crop_fracs=(0.0, 0.2), zooms=(1.0, 0.8, 1.2),
                 translate_fracs=(0.0, 0.1, -0.1),
                 metric="hybrid",
                 ecc_iter=100, ecc_eps=1e-5, alpha=0.0,
                 resume=False, fast_mode=True,
                 progress_callback=None):
    """
    Pipeline:
      1. Downsample target + tiles to coarse_width for candidate search.
      2. Try transforms (rot, flip, crop, zoom, translate).
      3. Rank candidates by metric.
      4. Hungarian assignment.
      5. Refine selected tiles on full-resolution with ECC.
    """
    t0 = time.time()
    target = load_image(target_path)
    target, h_step, w_step = center_crop_to_multiple(target, grid_n)
    H, W = target.shape[:2]
    N = grid_n * grid_n

    # ---------------- downsample target ----------------
    if fast_mode and coarse_width and W > coarse_width:
        scale = coarse_width / float(W)
    else:
        scale = 1.0
    small_target = cv2.resize(target, (int(W*scale), int(H*scale)))
    small_h, small_w = small_target.shape[:2]
    small_h_step, small_w_step = small_h // grid_n, small_w // grid_n

    # ---------------- load tiles (small only for coarse stage) ----------------
    tile_paths = list_tile_paths(tiles_folder)
    if not tile_paths:
        raise FileNotFoundError("No tiles found in folder")
    tiles_small = []
    for p in tile_paths:
        img = cv2.imread(p)
        if img is None: continue
        img_small = cv2.resize(img, (small_w_step, small_h_step))
        tiles_small.append((p, img_small))
    M = len(tiles_small)
    if M == 0:
        raise FileNotFoundError("No readable tiles")

    if progress_callback: progress_callback(2)

    # ---------------- coarse search ----------------
    candidates = [[] for _ in range(N)]
    total = N * M * len(rotations) * len(flips) * len(crop_fracs) * len(zooms) * len(translate_fracs)
    done = 0
    for i in range(grid_n):
        for j in range(grid_n):
            pidx = i*grid_n + j
            y0, y1 = i*small_h_step, (i+1)*small_h_step
            x0, x1 = j*small_w_step, (j+1)*small_w_step
            patch_small = small_target[y0:y1, x0:x1]
            best = []
            for t_idx, (tpath, t_small) in enumerate(tiles_small):
                for rot in rotations:
                    for flip in flips:
                        for crop in crop_fracs:
                            for zoom in zooms:
                                for tx in translate_fracs:
                                    for ty in translate_fracs:
                                        trans = transform_tile(
                                            t_small, patch_small.shape[:2],
                                            rotation=rot, flip=flip,
                                            crop_frac=crop, zoom=zoom,
                                            translate=(tx,ty)
                                        )
                                        # metrics
                                        ssimv = compute_ssim(patch_small, trans)
                                        labv = compute_lab_dist(patch_small, trans)
                                        mse = float(((patch_small.astype("float32") - trans.astype("float32"))**2).mean())
                                        if metric == "ssim":
                                            score = ssimv
                                        elif metric == "lab":
                                            score = -labv
                                        elif metric == "mse":
                                            score = -mse
                                        elif metric == "hybrid":
                                            score = ssimv - 0.3*(labv/100.0)
                                        else:
                                            score = ssimv

                                        if len(best) < top_k:
                                            best.append((score, t_idx, rot, flip, crop, zoom, tx, ty))
                                            best.sort(reverse=True)
                                        else:
                                            if score > best[-1][0]:
                                                best[-1] = (score, t_idx, rot, flip, crop, zoom, tx, ty)
                                                best.sort(reverse=True)

                                        done += 1
                                        if progress_callback and done % 1000 == 0:
                                            progress_callback(2 + int(30 * done / total))

            for (s, t_idx, rot, flip, crop, zoom, tx, ty) in best:
                candidates[pidx].append({
                    "tile_index": int(t_idx), "rotation": int(rot), "flip": bool(flip),
                    "crop_frac": float(crop), "zoom": float(zoom),
                    "translate": (float(tx), float(ty)),
                    "coarse_score": float(s)
                })

    if progress_callback: progress_callback(40)

    # ---------------- cost matrix + Hungarian ----------------
    big_cost = 10.0
    cost = np.ones((N, M)) * big_cost
    for pidx in range(N):
        for cand in candidates[pidx]:
            t_idx = cand["tile_index"]
            cost[pidx, t_idx] = min(cost[pidx, t_idx], 1.0 - cand["coarse_score"])
    rows, cols = linear_sum_assignment(cost)
    if progress_callback: progress_callback(55)

    # ---------------- refinement stage (load full tiles here only) ----------------
    mosaic = np.zeros_like(target)
    records = []
    for k, r in enumerate(rows):
        c = int(cols[k])
        cand_list = [cand for cand in candidates[r] if int(cand["tile_index"]) == c]
        cand = max(cand_list, key=lambda x: x.get("coarse_score", -999)) if cand_list else None

        y0, y1 = (r // grid_n) * h_step, ((r // grid_n) + 1) * h_step
        x0, x1 = (r % grid_n) * w_step, ((r % grid_n) + 1) * w_step
        dst_patch = target[y0:y1, x0:x1]

        # load full-res tile here
        tile_full = cv2.imread(tile_paths[c])
        if tile_full is None:
            init = np.zeros_like(dst_patch)
        else:
            if cand is None:
                init = cv2.resize(tile_full, (dst_patch.shape[1], dst_patch.shape[0]))
            else:
                init = transform_tile(tile_full, dst_patch.shape[:2],
                                      rotation=cand["rotation"], flip=cand["flip"],
                                      crop_frac=cand["crop_frac"], zoom=cand["zoom"],
                                      translate=cand["translate"])
        aligned, ok, warp, score = ecc_refine(init, dst_patch, max_iter=ecc_iter, eps=ecc_eps)

        mosaic[y0:y1, x0:x1] = aligned
        records.append({
            "patch_index": int(r), "tile_index": int(c), "tile_path": os.path.basename(tile_paths[c]),
            "rotation": int(cand["rotation"]) if cand else 0,
            "flip": bool(cand["flip"]) if cand else False,
            "crop_frac": float(cand["crop_frac"]) if cand else 0.0,
            "zoom": float(cand["zoom"]) if cand else 1.0,
            "translate": cand["translate"] if cand else (0.0,0.0),
            "coarse_score": float(cand["coarse_score"]) if cand else 0.0,
            "ecc_success": bool(ok), "ecc_score": float(score),
            "warp": warp.tolist() if hasattr(warp,"tolist") else None
        })

        if progress_callback:
            progress_callback(55 + int(40 * (k+1)/len(rows)))

    if alpha > 0:
        mosaic = cv2.addWeighted(mosaic, 1-alpha, target, alpha, 0)

    save_mosaic(mosaic, out_image)
    save_params_json(records, out_params_json)
    save_params_csv(records, out_params_csv)

    if progress_callback: progress_callback(100)
    return {"mosaic_path": out_image, "params_json": out_params_json, "params_csv": out_params_csv,
            "time_s": time.time()-t0}
