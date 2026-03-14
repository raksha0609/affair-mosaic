import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.feature import canny


def to_gray(img):
    if img is None:
        return img
    if len(img.shape) == 2:
        # already grayscale
        return img
    if img.shape[2] == 1:
        return img[:, :, 0]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



def transform_tile(tile, target_shape,
                   rotation=0, flip=False,
                   crop_frac=0.0, zoom=1.0, translate=(0.0, 0.0)):

    th, tw = target_shape
    h, w = tile.shape[:2]

    if crop_frac > 0:
        dy = int(h * crop_frac / 2)
        dx = int(w * crop_frac / 2)
        if 2 * dy < h and 2 * dx < w:
            tile = tile[dy:h - dy, dx:w - dx]
            h, w = tile.shape[:2]

    if flip:
        tile = cv2.flip(tile, 1)

    if rotation % 360 != 0:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, 1.0)
        tile = cv2.warpAffine(tile, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    if zoom != 1.0:
        tile = cv2.resize(tile, (int(w * zoom), int(h * zoom)))

    tx = int(translate[0] * w)
    ty = int(translate[1] * h)
    if tx or ty:
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        tile = cv2.warpAffine(tile, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return cv2.resize(tile, (tw, th))


def ecc_refine(src, dst, max_iter=100, eps=1e-5, motion=cv2.MOTION_EUCLIDEAN):
    try:
        src_g = to_gray(src)
        dst_g = to_gray(dst)
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
        _, warp = cv2.findTransformECC(dst_g, src_g, warp, motion, criteria)
        aligned = cv2.warpAffine(
            src, warp,
            (dst.shape[1], dst.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT
        )
        score = ssim(dst_g, to_gray(aligned),
                     data_range=dst_g.max() - dst_g.min() + 1e-6)
        return aligned, True, warp, float(score)
    except:
        return cv2.resize(src, (dst.shape[1], dst.shape[0])), False, None, -1.0


def compute_ssim(a, b):
    return float(ssim(to_gray(a), to_gray(b), data_range=255))


def compute_mse(a, b):
    return float(np.mean((a.astype("float32") - b.astype("float32")) ** 2))


def compute_lab_dist(a, b):
    la = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
    lb = cv2.cvtColor(b, cv2.COLOR_BGR2LAB)
    return float(np.mean((la - lb) ** 2) ** 0.5)


def compute_ncc(a, b):
    a = to_gray(a).astype("float32").ravel()
    b = to_gray(b).astype("float32").ravel()
    a -= a.mean()
    b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def compute_edge_ncc(a, b, sigma=2.0):
    ea = canny(to_gray(a) / 255.0, sigma=sigma).ravel()
    eb = canny(to_gray(b) / 255.0, sigma=sigma).ravel()
    ea -= ea.mean()
    eb -= eb.mean()
    denom = np.linalg.norm(ea) * np.linalg.norm(eb)
    return 0.0 if denom == 0 else float(np.dot(ea, eb) / denom)


# ===================== NEW (PHASE 1) =====================
def edge_strip(img, side, width=6):
    h, w = img.shape[:2]
    if side == "left":
        return img[:, :width]
    if side == "right":
        return img[:, w - width:]
    if side == "top":
        return img[:width, :]
    if side == "bottom":
        return img[h - width:, :]
    raise ValueError("Invalid side")
