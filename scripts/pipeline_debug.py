# scripts/pipeline_debug.py
import os, cv2, json, numpy as np
from glob import glob
from mosaic.core import compute_ssim
from mosaic.io import load_image, save_mosaic

def list_tile_paths(folder):
    return sorted(glob(os.path.join(folder, "*.png")) + glob(os.path.join(folder, "*.jpg")))

def run_pipeline_debug(target_path, tiles_folder, grid_n=10, coarse_width=200, metric="ssim"):
    """
    Simple debug version: splits target into patches, finds best tile per patch (no registration)
    and saves intermediate info.
    """
    print("[INFO] Running AFFAIR Debug Mode...")
    os.makedirs("results/debug", exist_ok=True)

    target = load_image(target_path)
    H, W = target.shape[:2]
    h_step, w_step = H // grid_n, W // grid_n

    tile_paths = list_tile_paths(tiles_folder)
    print(f"[DEBUG] Loaded {len(tile_paths)} tiles")

    # Resize tiles
    tiles_resized = []
    for p in tile_paths:
        t = cv2.imread(p)
        if t is None:
            continue
        t = cv2.cvtColor(t, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(t)
        l = cv2.equalizeHist(l)
        t = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        t = cv2.resize(t, (w_step, h_step))
        tiles_resized.append((p, t))

    mosaic = np.zeros_like(target)
    records = []

    for i in range(grid_n):
        for j in range(grid_n):
            y0, y1 = i * h_step, (i + 1) * h_step
            x0, x1 = j * w_step, (j + 1) * w_step
            patch = target[y0:y1, x0:x1]

            best_score = -1
            best_tile = None

            for (path, timg) in tiles_resized:
                score = compute_ssim(patch, timg)
                if score > best_score:
                    best_score = score
                    best_tile = path

            if best_tile is not None:
                best_img = [t for (p, t) in tiles_resized if p == best_tile][0]
                mosaic[y0:y1, x0:x1] = best_img
                records.append({
                    "row": i,
                    "col": j,
                    "best_tile": os.path.basename(best_tile),
                    "score": float(best_score)
                })
                print(f"[DEBUG] Placed patch ({i},{j}) with score {best_score:.3f}")

    save_path = "results/debug/mosaic_debug.jpg"
    cv2.imwrite(save_path, mosaic)
    print(f"[INFO] Debug mosaic saved → {save_path}")

    with open("results/debug/mosaic_debug_info.json", "w") as f:
        json.dump(records, f, indent=2)
    print("[INFO] Tile placement data saved → results/debug/mosaic_debug_info.json")

    return {"mosaic_path": save_path, "records": records}
