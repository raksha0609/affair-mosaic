# mosaic/core.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ---------------- image prep ----------------
def center_crop_to_multiple(img, grid_n):
    """Crop image so both dimensions are divisible by grid_n"""
    H, W = img.shape[:2]
    Hc = (H // grid_n) * grid_n
    Wc = (W // grid_n) * grid_n
    y0, x0 = (H - Hc)//2, (W - Wc)//2
    return img[y0:y0+Hc, x0:x0+Wc], Hc//grid_n, Wc//grid_n

# ---------------- transforms ----------------
def transform_tile(tile, target_shape,
                   rotation=0, flip=False,
                   crop_frac=0.0, zoom=1.0, translate=(0.0,0.0)):
    """
    Apply crop, flip, rotate, zoom, translate to tile and resize to target shape.
    """
    th, tw = target_shape
    h, w = tile.shape[:2]

    # crop
    if crop_frac > 0:
        dy, dx = int(h*crop_frac), int(w*crop_frac)
        if h-2*dy > 0 and w-2*dx > 0:
            tile = tile[dy:h-dy, dx:w-dx]

    # flip
    if flip:
        tile = cv2.flip(tile, 1)

    # rotate
    if rotation != 0:
        M = cv2.getRotationMatrix2D((tile.shape[1]//2, tile.shape[0]//2), rotation, 1.0)
        tile = cv2.warpAffine(tile, M, (tile.shape[1], tile.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # zoom
    if zoom != 1.0:
        h, w = tile.shape[:2]
        new_h, new_w = int(h*zoom), int(w*zoom)
        tile = cv2.resize(tile, (new_w, new_h))
        # pad or crop
        if new_h < h or new_w < w:
            pad_y, pad_x = (h-new_h)//2, (w-new_w)//2
            tile = cv2.copyMakeBorder(tile, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x,
                                      cv2.BORDER_REFLECT)
        else:
            y0, x0 = (new_h-h)//2, (new_w-w)//2
            tile = tile[y0:y0+h, x0:x0+w]

    # translate
    tx, ty = int(translate[0]*tile.shape[1]), int(translate[1]*tile.shape[0])
    M = np.float32([[1,0,tx],[0,1,ty]])
    tile = cv2.warpAffine(tile, M, (tile.shape[1], tile.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # resize to match patch
    tile = cv2.resize(tile, (tw, th))
    return tile

# ---------------- ECC refinement ----------------
def ecc_refine(src, dst, max_iter=100, eps=1e-5):
    """Refine alignment with ECC on grayscale."""
    try:
        warp = np.eye(2,3,dtype=np.float32)
        cc, warp = cv2.findTransformECC(
            cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(src, cv2.COLOR_BGR2GRAY),
            warp, motionType=cv2.MOTION_EUCLIDEAN,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
        )
        aligned = cv2.warpAffine(src, warp, (dst.shape[1], dst.shape[0]),
                                 flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REFLECT)
        score = ssim(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY),
                     cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))
        return aligned, True, warp, score
    except Exception:
        return cv2.resize(src, (dst.shape[1], dst.shape[0])), False, None, -1

# ---------------- metrics ----------------
def compute_ssim(a, b):
    return ssim(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
