# AFFAIR Mosaic Reconstruction (Master Thesis)

This project implements **rigid image registration with global optimization** for mosaic-based image reconstruction.  
It is part of the AFFAIR framework developed during my Master Thesis.

---

## Features
- **Modular pipeline**
  * `mosaic/core.py` – transformations, metrics, ECC refinement  
  * `mosaic/pipeline.py` – main reconstruction pipeline  
  * `mosaic/io.py` – image and parameter I/O  
  * `mosaic/gui.py` – PyQt GUI  
  * `scripts/run_cli.py` – CLI entry point  

- **Transformations**
  * Rotation (0°, 90°, 180°, 270°)  
  * Horizontal flipping  
  * Cropping (fractional cutouts)  
  * Zooming (in/out)  
  * Translation (fractional shifts)  

- **Similarity metrics**
  * SSIM  
  * LAB color distance  
  * MSE  
  * Hybrid (SSIM – LAB)  

- **Optimization**
  * Downsampling for fast coarse search (default: 200 px width)  
  * Hungarian algorithm for global assignment  
  * ECC refinement on full-resolution tiles  

- **Results**
  * Mosaic image → `results/mosaics/…`  
  * Parameters (JSON + CSV) → `results/params/…`  

---

## Configuration

The project uses a `.env` file for configuration variables. Copy the provided `.env` file and adjust the values as needed.

Available configuration options:
- `DEBUG`: Enable debug mode (default: False)
- `DATA_DIR`: Path to data directory (default: data/)
- `RESULTS_DIR`: Path to results directory (default: results/)
- `DEFAULT_GRID`: Default grid size (default: 10)
- `DEFAULT_COARSE`: Default coarse resolution (default: 200)
- `DEFAULT_METRIC`: Default similarity metric (default: hybrid)
- `DEFAULT_TOPK`: Default top-k candidates (default: 5)

---

## Usage

### Command Line Interface
```bash
python -m scripts.run_cli \
  --target data/target/lion.png \
  --tiles data/tiles \
  --grid 10 \
  --coarse 200 \
  --topk 5 \
  --metric hybrid
