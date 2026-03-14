from scripts.pipeline_debug import run_pipeline_debug

if __name__ == "__main__":
    out = run_pipeline_debug(
        target_path="data/target/back.jpg",
        tiles_folder="data/tiles",
        grid_n=10,
        coarse_width=200,
        metric="hybrid"
    )

    print(f"[DONE] Debug mosaic: {out['mosaic_path']}")