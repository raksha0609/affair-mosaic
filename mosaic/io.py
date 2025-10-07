# mosaic/io.py
import os
import json
import csv
import cv2

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def save_mosaic(img, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, img)

def save_params_json(params, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # make numpy arrays serializable
    def default(o):
        try:
            import numpy as np
            if isinstance(o, np.ndarray):
                return o.tolist()
        except Exception:
            pass
        return str(o)
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2, default=default)

def save_params_csv(records, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if not records:
        with open(out_path, "w") as f:
            f.write("")  # empty file
        return
    keys = list(records[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            # convert lists/tuples to string for CSV
            r2 = {k: (",".join(map(str, v)) if isinstance(v, (list, tuple)) else v) for k, v in r.items()}
            w.writerow(r2)
