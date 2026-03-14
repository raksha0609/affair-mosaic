import argparse
from mosaic.pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AFFAIR mosaic reconstruction (CLI).")
    parser.add_argument("--target", required=True)
    parser.add_argument("--tiles", required=True)

    parser.add_argument("--grid", type=int, help="Legacy square grid")
    parser.add_argument("--grid-rows", type=int, default=None)
    parser.add_argument("--grid-cols", type=int, default=None)

    parser.add_argument("--coarse", type=int, default=200)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--metric", type=str, default="hybrid")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--project", type=str, default="results/project/affair_project.json")

    args = parser.parse_args()

    if args.grid_rows is None or args.grid_cols is None:
        grid_rows = grid_cols = args.grid or 10
    else:
        grid_rows, grid_cols = args.grid_rows, args.grid_cols

    res = run_pipeline(
        target_path=args.target,
        tiles_folder=args.tiles,
        out_image="results/mosaics/mosaic_cli_result.jpg",
        out_params_json="results/params/mosaic_cli_params.json",
        out_params_csv="results/params/mosaic_cli_params.csv",
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        coarse_width=args.coarse,
        top_k=args.topk,
        metric=args.metric,
        alpha=args.alpha,
        resume=args.resume,
        project_path=args.project,
        progress_callback=lambda v: print(f"\rProgress: {v:3d}%", end="", flush=True)
    )

    print(f"\n✅ Done: {res['mosaic_path']}")
