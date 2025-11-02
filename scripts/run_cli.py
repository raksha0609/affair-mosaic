import argparse
from mosaic.pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AFFAIR mosaic reconstruction (CLI).")
    parser.add_argument("--target", required=True, help="Path to target image")
    parser.add_argument("--tiles", required=True, help="Path to folder with tiles")
    parser.add_argument("--grid", type=int, default=10, help="Grid size (e.g., 10)")
    parser.add_argument("--coarse", type=int, default=200, help="Coarse width in pixels")
    parser.add_argument("--topk", type=int, default=5, help="Top-K coarse candidates per patch")
    parser.add_argument("--metric", type=str, default="hybrid", help="Metric: ssim | mse | lab | ncc | hybrid")
    parser.add_argument("--alpha", type=float, default=0.0, help="Blend alpha (0–1)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing project file")
    parser.add_argument("--project", type=str, default="results/project/affair_project.json", help="Project checkpoint path")

    args = parser.parse_args()

    print("[INFO] Running pipeline...")
    try:
        res = run_pipeline(
            target_path=args.target,
            tiles_folder=args.tiles,
            out_image="results/mosaics/mosaic_cli_result.jpg",
            out_params_json="results/params/mosaic_cli_params.json",
            out_params_csv="results/params/mosaic_cli_params.csv",
            grid_n=args.grid,
            coarse_width=args.coarse,
            top_k=args.topk,
            metric=args.metric,
            alpha=args.alpha,
            resume=args.resume,
            project_path=args.project,
            progress_callback=lambda v: print(f"\rProgress: {v}%", end="", flush=True)
        )
        print(f"\n✅ Done! Mosaic saved at {res['mosaic_path']}")
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
