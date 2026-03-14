import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# mosaic/verify_runtime.py
import cv2
import time
import json
import psutil
import argparse
import threading
from mosaic.pipeline import run_pipeline


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def format_bytes(num):
    """Human-readable memory size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


class PeakMemoryTracker:
    """
    Background thread that polls process RSS every `interval` seconds
    and records the true peak memory during a pipeline run.
    """
    def __init__(self, interval=0.2):
        self._process  = psutil.Process(os.getpid())
        self._interval = interval
        self._peak     = self._process.memory_info().rss
        self._baseline = self._peak
        self._running  = False
        self._thread   = None

    def start(self):
        self._baseline = self._process.memory_info().rss
        self._peak     = self._baseline
        self._running  = True
        self._thread   = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def _poll(self):
        while self._running:
            try:
                rss = self._process.memory_info().rss
                if rss > self._peak:
                    self._peak = rss
            except psutil.NoSuchProcess:
                break
            time.sleep(self._interval)

    @property
    def baseline(self):
        return self._baseline

    @property
    def peak(self):
        return self._peak

    @property
    def net_increase(self):
        return self._peak - self._baseline


# ─────────────────────────────────────────────
#  Main verification function
# ─────────────────────────────────────────────

def verify_runtime(target, tiles, grid=10, coarse=200, metric="hybrid"):
    print("=" * 70)
    print("[VERIFY] AFFAIR Runtime Verification")
    print("=" * 70)

    # ── Stage 1: Run pipeline with peak memory tracking ──────────────────
    tracker = PeakMemoryTracker(interval=0.2)
    tracker.start()

    t0 = time.time()
    res = run_pipeline(
        target_path=target,
        tiles_folder=tiles,
        out_image="results/mosaics/verify_runtime_result.jpg",
        out_params_json="results/params/verify_runtime_params.json",
        out_params_csv="results/params/verify_runtime_params.csv",
        grid_n=grid,
        coarse_width=coarse,
        top_k=5,
        metric=metric,
        progress_callback=lambda v: print(f"\rProgress: {v}%", end="", flush=True)
    )
    t_total = time.time() - t0

    tracker.stop()
    print("\n[INFO] Pipeline finished successfully.")

    # ── Stage 2: Parse output params ─────────────────────────────────────
    params_json = res["params_json"]
    if not os.path.exists(params_json):
        print("[WARN] No parameter file found.")
        return

    with open(params_json, "r") as f:
        data = json.load(f)

    ecc_success   = sum(1 for d in data if d.get("ecc_success"))
    total_patches = len(data)

    # ── Stage 3: Target image info ────────────────────────────────────────
    target_img = cv2.imread(target)
    H, W = target_img.shape[:2]
    downsample_scale = coarse / float(W) if W > coarse else 1.0

    # ── Stage 4: Print summary ────────────────────────────────────────────
    print("\n========== RUNTIME SUMMARY ==========")
    print(f"Target              : {os.path.basename(target)}  ({W}x{H})")
    print(f"Tiles folder        : {tiles}")
    print(f"Grid size           : {grid} x {grid}  ({total_patches} patches)")
    print(f"Downsample scale    : {downsample_scale:.3f}")
    print(f"Coarse width        : {coarse}px")
    print(f"Similarity metric   : {metric}")
    print(f"ECC success         : {ecc_success}/{total_patches}")
    print(f"Total runtime       : {t_total/60:.2f} min  ({t_total:.1f} s)")
    print(f"Memory at start     : {format_bytes(tracker.baseline)}")
    print(f"Peak memory (RSS)   : {format_bytes(tracker.peak)}")
    print(f"Net increase        : {format_bytes(tracker.net_increase)}")
    print("Results:")
    print(f"  Mosaic            : {res['mosaic_path']}")
    print(f"  Params JSON       : {res['params_json']}")
    print(f"  Params CSV        : {res['params_csv']}")
    print("=====================================")
    print("Verification complete.\n")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AFFAIR Runtime Verifier")
    parser.add_argument("--target",  required=True,         help="Path to target image")
    parser.add_argument("--tiles",   required=True,         help="Path to tiles folder")
    parser.add_argument("--grid",    type=int, default=10,  help="Grid size N (NxN)")
    parser.add_argument("--coarse",  type=int, default=200, help="Coarse search width (px)")
    parser.add_argument("--metric",  default="hybrid",      help="Similarity metric")
    args = parser.parse_args()

    verify_runtime(
        target=args.target,
        tiles=args.tiles,
        grid=args.grid,
        coarse=args.coarse,
        metric=args.metric
    )