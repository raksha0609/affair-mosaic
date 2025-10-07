# scripts/run_cli.py
import argparse
import os
from mosaic.pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--tiles", required=True)
    parser.add_argument("--grid", type=int, default=10)
    parser.add_argument("--coarse", type=int, default=200)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--metric", choices=["ssim","hybrid","mse","lab","ncc"], default="hybrid")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--crop", default="0.0,0.2", help="comma list")
    parser.add_argument("--zooms", default="1.0,0.8,1.2", help="comma list")
    parser.add_argument("--translate", default="0.0,0.1,-0.1", help="comma list")
    parser.add_argument("--alpha", type=float, default=0.0)
    args = parser.parse_args()

    crop_fracs = tuple(float(x) for x in args.crop.split(",") if x.strip()!="")
    zooms = tuple(float(x) for x in args.zooms.split(",") if x.strip()!="")
    translate_fracs = tuple(float(x) for x in args.translate.split(",") if x.strip()!="")

    os.makedirs("results/mosaics", exist_ok=True)
    os.makedirs("results/params", exist_ok=True)

    print("[INFO] Running pipeline...")
    res = run_pipeline(args.target, args.tiles,
                       "results/mosaics/mosaic_cli.jpg",
                       "results/params/mosaic_cli_params.json",
                       "results/params/mosaic_cli_params.csv",
                       grid_n=args.grid, coarse_width=args.coarse, top_k=args.topk,
                       rotations=(0,90,180,270), flips=(False, True),
                       crop_fracs=crop_fracs, zooms=zooms, translate_fracs=translate_fracs,
                       metric=args.metric, ecc_iter=100, ecc_eps=1e-5,
                       alpha=args.alpha, resume=args.resume,
                       progress_callback=lambda v: print(f"\rProgress: {v}%", end="", flush=True))
    print(f"\n[INFO] Done in {res['time_s']:.1f}s. Mosaic saved to {res['mosaic_path']}")
