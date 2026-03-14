# AFFAIR — A Framework for Arbitrary Image Registration

**Master Thesis Project** | Rigid image registration and mosaic reconstruction via a modular, research-oriented pipeline.

---

## Project Structure

```
mosaic/
├── core.py          – Transformations, similarity metrics, ECC refinement
├── pipeline.py      – Full reconstruction pipeline (coarse search → assignment → ECC)
├── io.py            – Image loading, saving, JSON/CSV parameter export
└── gui.py           – PyQt5 graphical interface

scripts/
├── lines_and_natural/
│   ├── generate_phantom_from_image.py  – Generate natural edge-based phantom from a real image
│   └── generate_phantom_lines.py       – Generate line-pattern phantom
├── baseline_runner.py            – Comparison baselines (random / greedy)
├── extract_tiles.py              – Extract grid tiles from any image
├── generate_dense_phantom.py     – Generate dense coloured-blob phantom with GT
├── heatmap_debug.py              – Visualize tile-match score heatmap
├── phantom_eval.py               – Evaluate pipeline output against phantom GT
├── phantom_plot.py               – Plot phantom evaluation metrics
├── phantoms.py                   – Generate basic dot phantom dataset
├── pipeline_debug.py             – Debug pipeline (no ECC, logs per-patch scores)
├── run_cli.py                    – Main CLI entry point
├── run_cli_debug.py              – Debug mode CLI (no registration, fast inspection)
└── verify_runtime.py             – Runtime and memory profiling

synthetic_eval/
└── synthetic_eval_scripts/
    └── synthetic_mosaic.py       – Generate synthetic mosaic visualisations for thesis
                                    (reference, grid overlay, tile montage, error heatmap,
                                    alpha overlay, edge overlay)
```

---

## Pipeline Overview

The reconstruction pipeline operates in three sequential stages:

### 1. Coarse Candidate Search
- The target image and all tiles are downsampled (default: 200 px width)
- Every patch × tile × transformation hypothesis is scored with the selected similarity metric
- The top-K candidates per patch are retained (minimum 50 enforced as floor)
- Results are saved to a project file for optional resume (`--resume`)

### 2. Global Assignment (Hungarian Algorithm)
- A cost matrix is built from coarse scores + spatial regularisation penalties
- `scipy.optimize.linear_sum_assignment` solves the optimal one-to-one tile assignment
- Assignment is iterated up to 4 times, refining penalties using neighbour context
- Penalties applied:
  - **Position prior** — penalises intensity mismatch relative to expected vertical gradient
  - **Global rotation** — penalises deviation from the dominant rotation across the mosaic
  - **Local rotation** — penalises rotation discontinuity with left/top neighbours
  - **Flip consistency** — penalises flip-mode mismatch with neighbours
  - **Translation smoothness** — L1 penalty on translation offset differences
  - **Edge continuity** — edge-NCC between adjacent tile boundary strips (6 px wide)

### 3. ECC Refinement
- Each assigned tile is transformed at full resolution using the best candidate parameters
- `cv2.findTransformECC` (Euclidean motion model) refines the alignment
- Final mosaic and per-patch parameters are saved

---

## Transformation Hypotheses

The coarse search evaluates all combinations of:

| Parameter | Default values |
|-----------|---------------|
| Rotation | 0°, 90°, 180°, 270° |
| Flip | none, horizontal |
| Crop fraction | 0.0 |
| Zoom | 1.0 |
| Translation (x, y) | −0.05, 0.0, +0.05 (fractional) |

When the total hypothesis count exceeds `max_coarse_ops` (3 000 000), translations are symmetrically subsampled to keep coarse search tractable.

---

## Similarity Metrics

| Flag | Formula |
|------|---------|
| `hybrid` *(default)* | `0.45·SSIM + 0.25·NCC + 0.30·GradNCC − 0.10·(ΔE_LAB/100)` |
| `ssim` | Structural Similarity Index |
| `ncc` | Normalised Cross-Correlation |
| `mse` | Mean Squared Error (negated for maximisation) |
| `lab` | CIELAB colour distance (negated) |

Additional metrics used internally:
- **GradNCC** — NCC on Sobel gradient images
- **EdgeNCC** — NCC on Canny edge maps (used for boundary continuity penalty)

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** `numpy`, `opencv-python`, `scikit-image`, `scipy`, `pandas`, `pyqt5`, `psutil`

---

## Usage

### GUI
```bash
python -m mosaic.gui
```

### CLI — Standard Run
```bash
python -m scripts.run_cli \
  --target  data/target/image.png \
  --tiles   data/tiles \
  --grid    10 \
  --coarse  200 \
  --topk    5 \
  --metric  hybrid
```

Separate row/column grid control:
```bash
python -m scripts.run_cli \
  --target data/target/image.png \
  --tiles  data/tiles \
  --grid-rows 8 --grid-cols 12
```

Resume an interrupted run:
```bash
python -m scripts.run_cli --target ... --tiles ... --resume
```

### CLI Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--target` | *(required)* | Path to the reference/target image |
| `--tiles` | *(required)* | Directory of candidate tile images |
| `--grid` | `10` | Square grid size (N×N) |
| `--grid-rows` | — | Separate row count (overrides `--grid`) |
| `--grid-cols` | — | Separate column count (overrides `--grid`) |
| `--coarse` | `200` | Width (px) used for downsampled coarse search |
| `--topk` | `5` | Top-K candidates retained per patch |
| `--metric` | `hybrid` | Similarity metric: `hybrid`, `ssim`, `ncc`, `mse`, `lab` |
| `--alpha` | `0.0` | Blend original target over mosaic (0 = off) |
| `--resume` | off | Resume from saved project file |
| `--project` | `results/project/affair_project.json` | Project save/load path |

---

## Baselines

Two comparison baselines are provided for evaluation:

```bash
# Random assignment
python -m scripts.baseline_runner \
  --target data/target/image.png \
  --tiles  data/tiles \
  --mode   random --grid 6

# Greedy (best-per-patch, no global assignment)
python -m scripts.baseline_runner \
  --target data/target/image.png \
  --tiles  data/tiles \
  --mode   greedy --grid 6 --metric hybrid
```

---

## Phantom Evaluation

Three synthetic phantom types are supported, each targeting a different structural regime:

| Phantom | GT type | Primary metrics |
|---------|---------|----------------|
| `dots` | Parametric (rotation, translation, zoom) | Rotation error, translation error, zoom error |
| `lines` | Structural (orientation) | Orientation error (°), SSIM, NCC |
| `natural` | Perceptual only | ECC success rate, SSIM, NCC |

### Generate Phantoms

```bash
# Dense dot phantom (coloured blobs, full parametric GT)
python -m scripts.generate_dense_phantom

# Line phantom
python -m scripts.generate_phantom_lines

# Natural edge phantom (from a real image)
python -m scripts.generate_phantom_from_image --image path/to/image.jpg

# Basic dot phantom
python -m scripts.phantoms
```

### Extract Tiles from Phantom

```bash
python -m scripts.extract_tiles \
  --input  data/phantoms/targets/phantom_target.png \
  --output data/phantoms/tiles \
  --grid   6
```

### Run Evaluation

```bash
python -m scripts.phantom_eval \
  --phantom-type dots \
  --target  data/phantoms/targets/phantom_target.png \
  --mosaic  results/mosaics/mosaic_cli_result.jpg \
  --res     results/params/mosaic_cli_params.json \
  --gt      data/phantoms/targets/phantom_gt.json \
  --out     results/phantom_eval_results.csv

# Plot results
python -m scripts.phantom_plot
```

---

## Runtime Verification

```bash
python -m scripts.verify_runtime \
  --target data/target/image.png \
  --tiles  data/tiles \
  --grid   10 \
  --coarse 200 \
  --metric hybrid
```

Prints a full summary: grid size, ECC success rate, total runtime, baseline/peak RSS memory usage.

---

## Outputs

| Path | Content |
|------|---------|
| `results/mosaics/` | Reconstructed mosaic images |
| `results/params/` | Per-patch parameters (JSON + CSV) |
| `results/project/` | Saved coarse-search state for resume |
| `results/baselines/` | Baseline mosaic and params |
| `results/debug/` | Debug mosaic, per-patch scores JSON, heatmap PNG |
| `results/plots/` | Phantom evaluation metric plots |

---

## Thesis Context

Developed for the Master's thesis:

> **AFFAIR — A Framework for Arbitrary Image Registration with Focus on Rigid Registration**

Research objectives:
- Modular, configurable rigid registration pipeline
- Controlled evaluation using synthetic phantoms (dots, lines, natural)
- Analysis of similarity metrics, spatial regularisation, and global vs. local assignment
- Comparison against random and greedy baselines
- Runtime and memory characterisation

This is a **research framework**, not a production stitching tool.

---

## License

Provided for research and educational purposes.