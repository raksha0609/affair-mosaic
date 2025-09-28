# AFFAIR Mosaic Reconstruction (Master Thesis)

This project implements **rigid image registration with global optimization** 
for mosaic-based reconstruction. It is the Python reimplementation of the 
TurboReg-based pipeline, extended with AI-driven similarity measures.

---

## Features
- Modular pipeline (`mosaic/core.py`, `mosaic/pipeline.py`, `mosaic/io.py`)
- CLI and PyQt GUI interfaces
- Transformations:
  - Rotation (0°, 90°, 180°, 270°)
  - Flipping
  - Cropping
  - Zooming (in/out)
  - Translation (fractional)
- Multiple similarity metrics:
  - SSIM
  - LAB distance
  - MSE
  - Hybrid (SSIM – LAB)
- Downsampling for fast coarse registration
- Hungarian optimization for global assignment
- ECC refinement on full resolution
- Results saved as:
  - Mosaic image (`results/mosaics/…`)
  - Parameters JSON + CSV (`results/params/…`)
