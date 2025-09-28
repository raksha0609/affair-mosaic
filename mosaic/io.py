# mosaic/io.py
import os, cv2, json, csv

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img

def save_mosaic(img, path):
    cv2.imwrite(path, img)

def save_params_json(records, path):
    with open(path,"w") as f:
        json.dump(records,f,indent=2)

def save_params_csv(records, path):
    if not records: return
    keys = list(records[0].keys())
    with open(path,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow(r)
