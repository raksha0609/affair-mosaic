# mosaic/io.py
import os, json, csv, cv2

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
        open(out_path, "w").close()
        return
    keys = list(records[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            r2 = {k: (",".join(map(str, v)) if isinstance(v, (list, tuple)) else v) for k, v in r.items()}
            w.writerow(r2)

# --- Project / Resume helpers ---
def save_project(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_project(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)
