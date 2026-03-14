# scripts/heatmap_debug.py
import matplotlib.pyplot as plt
import numpy as np
import json
import os

if not os.path.exists("results/debug/mosaic_debug_info.json"):
    raise FileNotFoundError("Run scripts.run_cli_debug first to create debug info JSON!")

data = json.load(open("results/debug/mosaic_debug_info.json"))
grid_n = int(np.sqrt(len(data)))  # assumes square grid
heatmap = np.zeros((grid_n, grid_n))

for d in data:
    heatmap[d["row"], d["col"]] = d["score"]

plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.colorbar(label='SSIM Score')
plt.title('Tile Match Heatmap')
plt.savefig("results/debug/heatmap_debug.png")
plt.close()

print("[INFO] Heatmap saved at results/debug/heatmap_debug.png")
