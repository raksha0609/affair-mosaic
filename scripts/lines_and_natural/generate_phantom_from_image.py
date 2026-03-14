import os
import cv2
import argparse


def generate_phantom(img, sigma=5):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    edges = cv2.Canny(blur, 50, 150)
    phantom = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return phantom


def save_phantom(input_image):
    img = cv2.imread(input_image)
    if img is None:
        print(f"[ERROR] Cannot load image: {input_image}")
        return

    out_folder = "data/phantoms/natural"
    os.makedirs(out_folder, exist_ok=True)

    phantom = generate_phantom(img)
    out_path = os.path.join(out_folder, "phantom_natural_target.png")
    cv2.imwrite(out_path, phantom)

    print(f"[OK] Natural phantom created from {input_image}")
    print(f"[SAVED] {out_path}")
    print("[NEXT] Run extract_tiles to create tiles.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", type=str, required=True, help="Path to natural image"
    )
    args = parser.parse_args()

    save_phantom(args.image)
