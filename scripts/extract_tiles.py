import os
import cv2
import argparse

def extract_tiles(input_path, output_folder, grid=6):
    os.makedirs(output_folder, exist_ok=True)

    img = cv2.imread(input_path)
    if img is None:
        print(f"[ERROR] Cannot load image: {input_path}")
        return

    H, W = img.shape[:2]
    h_step = H // grid
    w_step = W // grid

    count = 0
    for r in range(grid):
        for c in range(grid):
            y0, y1 = r * h_step, (r + 1) * h_step
            x0, x1 = c * w_step, (c + 1) * w_step

            tile = img[y0:y1, x0:x1]
            out_path = os.path.join(output_folder, f"tile_{count:03d}.png")
            cv2.imwrite(out_path, tile)
            count += 1

    print(f"[OK] Extracted {count} tiles to: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--grid", type=int, default=6)
    args = parser.parse_args()

    extract_tiles(args.input, args.output, args.grid)
