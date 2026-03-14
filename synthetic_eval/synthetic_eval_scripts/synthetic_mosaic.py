import cv2
import numpy as np
import os
import random


def draw_grid(img, grid):
    gh, gw = grid
    h, w = img.shape[:2]
    tile_h = h // gh
    tile_w = w // gw
    grid_img = img.copy()

    for i in range(1, gh):
        y = i * tile_h
        cv2.line(grid_img, (0, y), (w, y), (255,255,255), 1)

    for j in range(1, gw):
        x = j * tile_w
        cv2.line(grid_img, (x, 0), (x, h), (255,255,255), 1)

    return grid_img


def create_tile_montage(tiles, rows=3, cols=3, tile_size=(128,128)):
    montage = np.zeros((rows*tile_size[1], cols*tile_size[0], 3), dtype=np.uint8)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= len(tiles):
                break
            t = cv2.resize(tiles[idx], tile_size)
            y1 = i * tile_size[1]
            y2 = y1 + tile_size[1]
            x1 = j * tile_size[0]
            x2 = x1 + tile_size[0]
            montage[y1:y2, x1:x2] = t
            idx += 1
    return montage


def edge_overlay(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    e1 = cv2.Canny(g1, 50, 150)
    e2 = cv2.Canny(g2, 50, 150)

    overlay = np.zeros_like(img1)
    overlay[:,:,1] = e1  # green
    overlay[:,:,2] = e2  # red
    return overlay


def alpha_overlay(img1, img2, alpha=0.5):
    return cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)


# -----------------------------
# Core Generator
# -----------------------------

def generate_synthetic_mosaic_full(
    img_path,
    name,
    out_dir,
    grid=(8,6),
    rot_deg=2,
    trans_px=3,
    crop_frac=0.05,
    tile_samples=9
):
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    gh, gw = grid
    tile_h = h // gh
    tile_w = w // gw

    recon = np.zeros_like(img)
    warped_tiles = []


    for i in range(gh):
        for j in range(gw):
            y1 = i * tile_h
            y2 = (i+1) * tile_h
            x1 = j * tile_w
            x2 = (j+1) * tile_w

            tile = img[y1:y2, x1:x2].copy()

            angle = random.uniform(-rot_deg, rot_deg)
            tx = random.randint(-trans_px, trans_px)
            ty = random.randint(-trans_px, trans_px)

            ch, cw = tile.shape[:2]
            M = cv2.getRotationMatrix2D((cw//2, ch//2), angle, 1.0)
            M[:,2] += [tx, ty]

            warped = cv2.warpAffine(tile, M, (cw, ch), borderMode=cv2.BORDER_REFLECT)

            # crop
            cf = crop_frac
            cy1 = int(cf * ch)
            cy2 = int((1-cf) * ch)
            cx1 = int(cf * cw)
            cx2 = int((1-cf) * cw)
            warped = warped[cy1:cy2, cx1:cx2]
            warped = cv2.resize(warped, (cw, ch))

            recon[y1:y2, x1:x2] = warped

            if len(warped_tiles) < tile_samples:
                warped_tiles.append(warped)

    cv2.imwrite(f"{out_dir}/{name}_reference.png", img)

    grid_img = draw_grid(img, grid)
    cv2.imwrite(f"{out_dir}/{name}_grid.png", grid_img)

    montage = create_tile_montage(warped_tiles)
    cv2.imwrite(f"{out_dir}/{name}_tile_montage.png", montage)

    cv2.imwrite(f"{out_dir}/{name}_reconstruction.png", recon)

    # error heatmap
    gray_ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_rec = cv2.cvtColor(recon, cv2.COLOR_BGR2GRAY)
    err = cv2.absdiff(gray_ref, gray_rec)
    err_color = cv2.applyColorMap(err, cv2.COLORMAP_JET)
    cv2.imwrite(f"{out_dir}/{name}_error.png", err_color)

    # overlay
    overlay = alpha_overlay(img, recon)
    cv2.imwrite(f"{out_dir}/{name}_overlay.png", overlay)

    # edge overlay
    eoverlay = edge_overlay(img, recon)
    cv2.imwrite(f"{out_dir}/{name}_edge_overlay.png", eoverlay)

    print(f"[OK] Generated all figures for {name}")


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    out_dir = "../thesis_figures"
    os.makedirs(out_dir, exist_ok=True)

    generate_synthetic_mosaic_full(
        "../input/dense_dot_reference.png",
        "dense_dot",
        out_dir
    )

    generate_synthetic_mosaic_full(
        "../input/line_reference.png",
        "line",
        out_dir
    )

    generate_synthetic_mosaic_full(
        "../input/natural_reference.png",
        "natural",
        out_dir
    )
