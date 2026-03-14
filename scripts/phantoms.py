# scripts/phantom.py
"""
Generate synthetic phantom data for AFFAIR evaluation.

Creates:
  - data/phantoms/tiles/*.png
  - data/phantoms/targets/phantom_target.png
  - data/phantoms/targets/phantom_gt.json
"""

import os
import cv2
import numpy as np
import random
import json

os.makedirs("data/phantoms/tiles", exist_ok=True)
os.makedirs("data/phantoms/targets", exist_ok=True)


def make_tile(w=200, h=200, seed=0):
    random.seed(seed)
    img = np.ones((h, w, 3), dtype="uint8") * 255

    # background noise
    for _ in range(300):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        img[y, x] = (
            random.randint(80, 255),
            random.randint(80, 255),
            random.randint(80, 255),
        )

    # draw primitives
    for _ in range(random.randint(1, 3)):
        color = tuple(random.randint(0, 255) for _ in range(3))
        cx = random.randint(20, w - 20)
        cy = random.randint(20, h - 20)
        r = random.randint(15, 50)
        cv2.circle(img, (cx, cy), r, color, -1)

    return img


# --- Generate tiles ---
for k in range(36):
    tile = make_tile(200, 200, seed=k)
    cv2.imwrite(f"data/phantoms/tiles/tile_{k:03d}.png", tile)

# --- Compose target ---
grid = 6
tile_files = sorted(os.listdir("data/phantoms/tiles"))
gt_info = {"patches": []}

tile_size = 200
target = np.zeros((grid * tile_size, grid * tile_size, 3), dtype="uint8")

for i in range(grid):
    for j in range(grid):
        idx = (i * grid + j) % len(tile_files)
        fname = tile_files[idx]
        tile = cv2.imread(os.path.join("data/phantoms/tiles", fname))

        angle = random.choice([0, 90, 180, 270])
        zoom = random.choice([0.9, 1.0, 1.1])

        M = cv2.getRotationMatrix2D(
            (tile_size // 2, tile_size // 2), angle, zoom
        )
        transformed = cv2.warpAffine(tile, M, (tile_size, tile_size))

        target[
            i * tile_size : (i + 1) * tile_size,
            j * tile_size : (j + 1) * tile_size,
        ] = transformed

        gt_info["patches"].append(
            {
                "patch_row": i,
                "patch_col": j,
                "tile_name": fname,
                "rotation": angle,
                "zoom": zoom,
                "translate": [0, 0],
            }
        )

cv2.imwrite("data/phantoms/targets/phantom_target.png", target)

with open("data/phantoms/targets/phantom_gt.json", "w") as f:
    json.dump(gt_info, f, indent=2)

print("[OK] Phantom generated")
print(" - Tiles: data/phantoms/tiles/")
print(" - Target: data/phantoms/targets/phantom_target.png")
print(" - GT: data/phantoms/targets/phantom_gt.json")
