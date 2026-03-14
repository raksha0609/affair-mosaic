import os
import cv2
import numpy as np
import random


def generate_line_phantom(width=800, height=800, num_lines=120):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    for _ in range(num_lines):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)

        color = (0, 0, 0)
        thickness = random.randint(1, 5)

        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    return img


def save_phantom():
    out_folder = "data/phantoms/lines"
    os.makedirs(out_folder, exist_ok=True)

    phantom = generate_line_phantom()
    out_path = os.path.join(out_folder, "phantom_lines_target.png")
    cv2.imwrite(out_path, phantom)

    print(f"[OK] Line phantom saved: {out_path}")
    print("[NEXT] Run extract_tiles to create tiles.")


if __name__ == "__main__":
    save_phantom()
