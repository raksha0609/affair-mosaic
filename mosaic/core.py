# mosaic/core.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def center_crop_to_multiple(img, grid_n):
    """Crop image so both dimensions are divisible by grid_n"""
    H, W = img.shape[:2]
    h_step = H // grid_n
    w_step = W // grid_n
    newH = h_step * grid_n
    newW = w_step * grid_n
    y0 = (H - newH) // 2
    x0 = (W - newW) // 2
    return img[y0:y0+newH, x0:x0+newW].copy(), h_step, w_step

def transform_tile(tile, target_shape,
                   rotation=0, flip=False,
                   crop_frac=0.0, zoom=1.0, translate=(0.0,0.0)):
    """
    Apply crop, flip, rotate, zoom, translate to tile and resize to target shape.
    target_shape is (h, w)
    translate is fraction of width/height (tx, ty)
    """
    th, tw = target_shape
    h, w = tile.shape[:2]
    # center crop
    if crop_frac and crop_frac > 0:
        dy = int(round(h * crop_frac / 2.0))
        dx = int(round(w * crop_frac / 2.0))
        if 2*dy < h and 2*dx < w:
            tile = tile[dy:h-dy, dx:w-dx]

    # flip horizontally
    if flip:
        tile = cv2.flip(tile, 1)

    # rotate in-place around center
    if rotation % 360 != 0:
        Mrot = cv2.getRotationMatrix2D((tile.shape[1]//2, tile.shape[0]//2), rotation, 1.0)
        tile = cv2.warpAffine(tile, Mrot, (tile.shape[1], tile.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # zoom (scale about center)
    if zoom != 1.0:
        h0, w0 = tile.shape[:2]
        new_h = max(1, int(round(h0 * zoom)))
        new_w = max(1, int(round(w0 * zoom)))
        tile = cv2.resize(tile, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # pad/crop back to original tile size before final resize
        if new_h < h0 or new_w < w0:
            pad_top = (h0 - new_h) // 2
            pad_left = (w0 - new_w) // 2
            pad_bottom = h0 - new_h - pad_top
            pad_right = w0 - new_w - pad_left
            tile = cv2.copyMakeBorder(tile, pad_top, pad_bottom, pad_left, pad_right,
                                      borderType=cv2.BORDER_REFLECT)
        else:
            # crop center
            y0 = (new_h - h0) // 2
            x0 = (new_w - w0) // 2
            tile = tile[y0:y0+h0, x0:x0+w0]

    # translate (fractions of tile width/height)
    tx = int(round(translate[0] * tile.shape[1]))
    ty = int(round(translate[1] * tile.shape[0]))
    if tx != 0 or ty != 0:
        Mtrans = np.float32([[1, 0, tx], [0, 1, ty]])
        tile = cv2.warpAffine(tile, Mtrans, (tile.shape[1], tile.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # final resize to requested patch size
    tile = cv2.resize(tile, (tw, th), interpolation=cv2.INTER_LINEAR)
    return tile

def ecc_refine(src, dst, max_iter=100, eps=1e-5, motion=cv2.MOTION_EUCLIDEAN):
    """
    Run ECC to refine alignment from src -> dst.
    Returns: aligned_color, success(bool), warp_matrix (2x3 or None), final_score (SSIM)
    """
    try:
        src_gray = to_gray(src)
        dst_gray = to_gray(dst)
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
        cc, warp = cv2.findTransformECC(dst_gray, src_gray, warp, motionType=motion, criteria=criteria)
        aligned = cv2.warpAffine(src, warp, (dst.shape[1], dst.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REFLECT)
        # compute SSIM (robust to constant images)
        try:
            dr = float(dst_gray.max() - dst_gray.min()) if dst_gray.max() != dst_gray.min() else 1.0
            s = ssim(dst_gray, cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), data_range=dr)
        except Exception:
            s = -1.0
        return aligned, True, warp, float(s)
    except Exception:
        # fallback: resize src to dst and return fail flag
        return cv2.resize(src, (dst.shape[1], dst.shape[0])), False, None, -1.0

# metrics helpers
def compute_ssim(a, b):
    a_gray = to_gray(a)
    b_gray = to_gray(b)
    dr = float(b_gray.max() - b_gray.min()) if b_gray.max() != b_gray.min() else 1.0
    try:
        return float(ssim(a_gray, b_gray, data_range=dr))
    except Exception:
        return -1.0

def compute_mse(a, b):
    return float(np.mean((a.astype("float32") - b.astype("float32"))**2))

def compute_lab_dist(a, b):
    la = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype("float32")
    lb = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype("float32")
    return float(np.mean((la - lb)**2)**0.5)

def compute_ncc(a, b):
    # normalized cross-correlation on grayscale vectors
    ag = to_gray(a).astype("float32").ravel()
    bg = to_gray(b).astype("float32").ravel()
    ag = (ag - ag.mean())
    bg = (bg - bg.mean())
    denom = (np.linalg.norm(ag) * np.linalg.norm(bg))
    if denom == 0:
        return 0.0
    return float(np.dot(ag, bg) / denom)
