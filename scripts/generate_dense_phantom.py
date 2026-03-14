import os
import cv2
import numpy as np
import random
import json

# -------------------- Paths --------------------
phantom_target_path = "data/phantoms/targets/phantom_target.png"
gt_path = "data/phantoms/targets/phantom_gt.json"
tiles_folder = "data/phantoms/tiles"

os.makedirs(os.path.dirname(phantom_target_path), exist_ok=True)
os.makedirs(tiles_folder, exist_ok=True)

# -------------------- Target generation --------------------
H, W = 600, 600
target = np.ones((H, W, 3), np.uint8) * 255

# Large colored blobs
for _ in range(200):
    radius = random.randint(10, 40)
    color = tuple(np.random.randint(0, 255, size=3).tolist())
    center = (
        random.randint(radius, W - radius),
        random.randint(radius, H - radius)
    )
    cv2.circle(target, center, radius, color, -1)

# Small dense dots
for _ in range(500):
    r = random.randint(2, 10)
    color = tuple(np.random.randint(0, 255, size=3).tolist())
    center = (
        random.randint(r, W - r),
        random.randint(r, H - r)
    )
    cv2.circle(target, center, r, color, -1)

cv2.imwrite(phantom_target_path, target)
print(f"[OK] Saved dense phantom target: {phantom_target_path}")

# -------------------- Tile extraction + GT --------------------
tile_size = 100
rows = H // tile_size
cols = W // tile_size

gt = {"patches": []}
tile_idx = 0

for r in range(rows):
    for c in range(cols):
        y0 = r * tile_size
        x0 = c * tile_size
        patch = target[y0:y0 + tile_size, x0:x0 + tile_size]

        # --- Forward transform (used to CREATE tile) ---
        angle = random.uniform(-10.0, 10.0)
        scale = random.uniform(0.9, 1.1)
        tx = random.uniform(-5.0, 5.0)
        ty = random.uniform(-5.0, 5.0)

        center = (tile_size // 2, tile_size // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        tile = cv2.warpAffine(
            patch,
            M,
            (tile_size, tile_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        tile_name = f"tile_{tile_idx:03d}.png"
        cv2.imwrite(os.path.join(tiles_folder, tile_name), tile)

        # --- INVERSE transform = what pipeline must estimate ---
        inv_scale = 1.0 / scale
        inv_angle = -angle

        inv_tx = -tx * inv_scale
        inv_ty = -ty * inv_scale

        gt["patches"].append({
            "patch_row": r,
            "patch_col": c,
            "tile_name": tile_name,
            "rotation": inv_angle,
            "zoom": inv_scale,
            "translate": [inv_tx, inv_ty]
        })

        tile_idx += 1

with open(gt_path, "w") as f:
    json.dump(gt, f, indent=2)

print(f"[OK] Generated {tile_idx} tiles")
print(f"[OK] Ground truth saved to: {gt_path}")
