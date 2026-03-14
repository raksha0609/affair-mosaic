"""
Compare pipeline output with phantom ground truth (if available).

Phantom types:
- dots    → full GT-based evaluation (rotation / translation / zoom)
- lines   → geometry-based evaluation (edge orientation consistency)
- natural → structure-only evaluation (ECC success + SSIM)

This avoids invalid metrics and NaNs while remaining mathematically correct.
"""

import os
import json
import cv2
import numpy as np
import argparse
from skimage.metrics import structural_similarity as ssim


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def dominant_edge_angle(gray):
    """
    Estimate dominant edge orientation using Hough transform.
    Returns angle in degrees, or None if no strong lines detected.
    """
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
    if lines is None:
        return None
    angles = [l[0][1] for l in lines]  # radians
    return np.mean(angles) * 180.0 / np.pi


def safe_mean(arr):
    """Return mean or None if array empty."""
    return float(np.mean(arr)) if len(arr) > 0 else None

def compute_ncc(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / denom) if denom != 0 else 0.0


def compute_mse(a, b):
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def compute_lab_delta(a_bgr, b_bgr):
    lab_a = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2LAB)
    lab_b = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2LAB)
    return float(np.mean(np.linalg.norm(lab_a - lab_b, axis=2)))

# ------------------------------------------------------------
# Core evaluation
# ------------------------------------------------------------

def compare_phantom(gt_path, res_path, target_img, mosaic_img, out_csv, phantom_type):
    # --- Load results
    with open(res_path, "r") as f:
        res = json.load(f)

    # --- Load images
    gt_img = cv2.imread(target_img)
    mo_img = cv2.imread(mosaic_img)

    if gt_img is None or mo_img is None:
        raise FileNotFoundError("Target or mosaic image not found")

    gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
    mo_gray = cv2.cvtColor(mo_img, cv2.COLOR_BGR2GRAY)

    if gt_gray.shape != mo_gray.shape:
        mo_gray = cv2.resize(mo_gray, (gt_gray.shape[1], gt_gray.shape[0]))

    ssim_score = ssim(gt_gray, mo_gray)
    
    ncc_score = compute_ncc(gt_gray, mo_gray)
    mse_score = compute_mse(gt_gray, mo_gray)
    lab_score = compute_lab_delta(gt_img, mo_img)

    # --- ECC success
    ecc_success = sum(1 for r in res if r.get("ecc_success", False))
    ecc_ratio = ecc_success / max(1, len(res))

    # --------------------------------------------------------
    # DOTS PHANTOM (full GT)
    # --------------------------------------------------------
    if phantom_type == "dots":
        if not gt_path or not os.path.exists(gt_path):
            raise FileNotFoundError("Dots phantom requires GT JSON")

        with open(gt_path, "r") as f:
            gt = json.load(f)

        gt_map = {(p["patch_row"], p["patch_col"]): p for p in gt["patches"]}

        rot_err, trans_err, zoom_err = [], [], []

        for r in res:
            key = (r["patch_row"], r["patch_col"])
            if key not in gt_map:
                continue

            g = gt_map[key]

            # rotation
            re = abs(r.get("rotation", 0) - g.get("rotation", 0))
            re = min(re, 360 - re)
            rot_err.append(re)

            # translation
            gt_tx, gt_ty = g.get("translate", [0, 0])
            rx, ry = r.get("translate", [0, 0])
            trans_err.append(np.hypot(gt_tx - rx, gt_ty - ry))

            # zoom
            zoom_err.append(abs(r.get("zoom", 1.0) - g.get("zoom", 1.0)))

        mean_rot = safe_mean(rot_err)
        mean_trans = safe_mean(trans_err)
        mean_zoom = safe_mean(zoom_err)

    # --------------------------------------------------------
    # LINES PHANTOM (orientation consistency)
    # --------------------------------------------------------
    elif phantom_type == "lines":
        gt_angle = dominant_edge_angle(gt_gray)
        mo_angle = dominant_edge_angle(mo_gray)

        mean_rot = abs(gt_angle - mo_angle) if (gt_angle is not None and mo_angle is not None) else None
        mean_trans = None
        mean_zoom = None

    # --------------------------------------------------------
    # NATURAL PHANTOM (structure only)
    # --------------------------------------------------------
    elif phantom_type == "natural":
        mean_rot = None
        mean_trans = None
        mean_zoom = None

    else:
        raise ValueError(f"Unknown phantom type: {phantom_type}")

    # --------------------------------------------------------
    # Reporting
    # --------------------------------------------------------
    print("\n📊 Phantom Evaluation Results")
    print("--------------------------------")

    if mean_rot is not None:
        print(f"Rotation Error (°): {mean_rot:.2f}")
    else:
        print("Rotation Error (°): N/A")

    if mean_trans is not None:
        print(f"Translation Error: {mean_trans:.4f}")
    else:
        print("Translation Error: N/A")

    if mean_zoom is not None:
        print(f"Zoom Error: {mean_zoom:.4f}")
    else:
        print("Zoom Error: N/A")

    print(f"ECC Success Rate: {ecc_ratio * 100:.1f}%")
    print(f"Overall SSIM: {ssim_score:.4f}")
    print(f"Overall NCC:  {ncc_score:.4f}")
    print(f"Overall MSE:  {mse_score:.4f}")
    print(f"LAB ΔE:       {lab_score:.4f}")


    # --------------------------------------------------------
    # CSV output (only if GT-based)
    # --------------------------------------------------------
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w") as f:
        f.write(
    "phantom_type,rotation_err,translation_err,zoom_err,"
    "ecc_success,ssim,ncc,mse,lab\n"
)
        f.write(
    f"{phantom_type},"
    f"{mean_rot if mean_rot is not None else ''},"
    f"{mean_trans if mean_trans is not None else ''},"
    f"{mean_zoom if mean_zoom is not None else ''},"
    f"{ecc_ratio:.4f},"
    f"{ssim_score:.4f},"
    f"{ncc_score:.4f},"
    f"{mse_score:.4f},"
    f"{lab_score:.4f}\n"
)


    print(f"[OK] Results saved to {out_csv}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phantom-type", choices=["dots", "lines", "natural"], required=True)
    parser.add_argument("--res", default="results/params/mosaic_cli_params.json")
    parser.add_argument("--target", required=True)
    parser.add_argument("--mosaic", default="results/mosaics/mosaic_cli_result.jpg")
    parser.add_argument("--gt", default=None)
    parser.add_argument("--out", default="results/phantom_eval_results.csv")

    args = parser.parse_args()

    compare_phantom(
        gt_path=args.gt,
        res_path=args.res,
        target_img=args.target,
        mosaic_img=args.mosaic,
        out_csv=args.out,
        phantom_type=args.phantom_type
    )
