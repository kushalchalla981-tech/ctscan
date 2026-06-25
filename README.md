<div align="center">

# 🧠 CT Reconstruction using LU Decomposition

<p align="center">
  <strong>From X-ray projections to image reconstruction —<br>a first-principles implementation in Python</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-1.24+-013243?style=flat&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/SciPy-1.10+-8CAAE6?style=flat&logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/pytest-7.x-0A9EDC?style=flat&logo=pytest&logoColor=white" alt="pytest">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat" alt="License">
  <img src="https://img.shields.io/badge/status-college%20project-FF6F00?style=flat" alt="Status">
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-cli-reference">CLI Reference</a> •
  <a href="#-results">Results</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#%EF%B8%8F-presentation">Presentation</a>
</p>

---

[![View Presentation](https://img.shields.io/badge/▶-View%20Presentation%20(HTML)-FF6F00?style=for-the-badge)](https://kushalchalla981-tech.github.io/ctscan/presentation.html)
[![Download PPTX](https://img.shields.io/badge/⬇-Download%20PPTX-2E77BC?style=for-the-badge)](presentation.pptx)
[![Open in Colab](https://img.shields.io/badge/▶-Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/kushalchalla981-tech/ctscan/blob/main/demo.ipynb)

</div>

## 📌 Overview

This project implements a **complete CT (Computed Tomography) reconstruction pipeline from first principles**. It demonstrates how a system of linear equations derived from X-ray projections can be solved using **LU decomposition** and **iterative LSQR** methods to recover cross-sectional images — the mathematical foundation behind every medical CT scanner.

> 🎓 **College project** · Linear Algebra · Medical Imaging · Inverse Problems

## ✨ Features

| Area | What it does |
|------|-------------|
| **🧪 Phantom Generation** | Shepp-Logan phantom (32×32, configurable size) with 10 anatomical ellipses |
| **📐 Ray Tracing** | Parallel-beam geometry — 44 angles × 32 detectors = 1,408 projection rays |
| **🧮 LU Decomposition** | Dense PA=LU with partial pivoting + forward/backward substitution |
| **⚡ Sparse LSQR** | Iterative solver via Golub-Kahan bidiagonalization for large sparse systems |
| **🔄 Iterative Refinement** | Correction loop `xₖ₊₁ = xₖ + A†·(b − Axₖ)` until residual < 10⁻¹⁰ |
| **🔧 Tikhonov Regularization** | Damped least-squares `min ‖Ax−b‖² + λ²‖x‖²` for noise robustness |
| **📊 Quality Metrics** | RMSE, PSNR, SSIM, relative error, forward residual |
| **📂 DICOM Support** | Load real CT scans (.dcm) via pydicom + pylibjpeg |
| **📝 HTML Reports** | All commands support `--html` for standalone self-contained reports |
| **📓 Jupyter Notebook** | Interactive walkthrough in `demo.ipynb` |
| **✅ 31 Unit Tests** | pytest suite across all 4 project phases |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Reconstruct the Shepp-Logan phantom
python main.py reconstruct

# With iterative refinement (near-perfect results)
python main.py reconstruct --refine

# Upload your own image (opens file dialog)
python main.py upload

# Menu-driven mode — no flags needed
python main.py interactive
```

## 📖 CLI Reference

| Command | Description |
|---------|-------------|
| `python main.py reconstruct [options]` | Full reconstruction pipeline (phantom or DICOM) |
| `python main.py upload [options]` | Pick an image → simulate CT reconstruction |
| `python main.py validate [--phase N] [--all]` | Run validation checks |
| `python main.py noise [options]` | Noise robustness sweep |
| `python main.py interactive` | Menu-driven interactive mode |
| `python main.py info` | Project information |

<details>
<summary><strong>🔍 Detailed CLI Flags</strong></summary>

### `reconstruct`

```bash
python main.py reconstruct --size 32 --refine --output result.png
python main.py reconstruct --input samples/CT-brain.dcm --compare --save-metrics results.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--size` | 32 | Phantom dimension (pixels) |
| `--refine` | off | Apply iterative refinement |
| `--method` | auto | Solver: `auto`, `sparse`, or `dense` |
| `--input` / `-i` | none | Path to DICOM or image file |
| `--compare` / `-c` | none | Save 4-panel comparison plot |
| `--save-metrics` / `-m` | none | Export metrics to JSON |
| `--output` / `-o` | none | Save final plot to file |
| `--html` | off | Export standalone HTML report |

### `validate`

```bash
python main.py validate --all
python main.py validate --phase 2
```

| Flag | Description |
|------|-------------|
| `--phase` 1–4 | Run a specific validation phase |
| `--all` | Run all validations |
| `--size` | Phantom size (default: 32) |
| `--html` | Export standalone HTML report |

### `noise`

```bash
python main.py noise --levels 0 1 5 10 20 --regularize --plot noise.png
```

| Flag | Default | Description |
|------|---------|-------------|
| `--levels` | 0 1 5 10 20 | Noise levels in percent |
| `--regularize` | off | Enable Tikhonov regularization |
| `--damp` | 2.0 | Regularization strength (√λ) |
| `--size` | 32 | Phantom size |
| `--html` | off | Export standalone HTML report |

</details>

## 📊 Results

### Without Noise (32×32)

| Metric | Basic | With Refinement |
|--------|:-----:|:---------------:|
| **RMSE** ↓ | 0.0223 | **0.0012** |
| **PSNR** ↑ | 33.38 dB | **58.35 dB** |
| **SSIM** ↑ | 0.995 | **1.000** |
| **Residual** ↓ | 2.6×10⁻⁵ | **2.3×10⁻⁷** |

> ✨ **SSIM = 1.000** with iterative refinement: the reconstruction is structurally indistinguishable from the original phantom.

### Noise Robustness (Tikhonov Regularization, damp=2.0)

| Noise | RMSE ↓ | PSNR ↑ | SSIM ↑ |
|:-----:|:------:|:------:|:------:|
| 0% | 0.083 | 21.6 dB | 0.913 |
| 1% | **0.095** | **20.4 dB** | **0.897** |
| 5% | 0.245 | 14.7 dB | 0.551 |
| 10% | 0.467 | 12.1 dB | 0.252 |
| 20% | 0.922 | 10.6 dB | 0.080 |

> ⚠️ Without regularization, RMSE jumps from 0.022 → **2.56** at just 1% noise. Regularization keeps it at **0.095** — **27× better**.

## 🏗 Architecture

```
main.py                    ← Single entry point (argparse + rich CLI)
│
└── src/
    ├── phantom.py         Shepp-Logan phantom generator (10 ellipses)
    ├── projector.py       Parallel-beam ray tracer → sparse A + sinogram b
    ├── lud_solver.py      LU decomposition with partial pivoting + LSQR
    ├── reconstructor.py   Full reconstruction pipeline orchestrator
    ├── fbp_solver.py      Filtered Back Projection (Radon/iradon)
    ├── metrics.py         RMSE, PSNR, SSIM, relative error, residual
    ├── noise.py           Gaussian/Poisson noise + robustness testing
    ├── exporter.py        Standalone HTML report generator
    ├── loader.py          DICOM/raster image loader (pydicom)
    ├── validate.py        4-phase consolidated validation suite
    ├── phantom3d.py       3D Shepp-Logan with evolving anatomy
    └── reconstructor3d.py 3D volume reconstruction pipeline
│
├── tests/                 31 unit tests (pytest)
├── samples/               73+ test images (DICOM + PNG)
├── demo.ipynb             Jupyter notebook walkthrough
└── presentation.html      Interactive HTML slide deck
```

## ▶️ Presentation

View the full project presentation:

| Format | Link |
|--------|------|
| 🌐 **HTML** (interactive) | [kushalchalla981-tech.github.io/ctscan/presentation.html](https://kushalchalla981-tech.github.io/ctscan/presentation.html) |
| 📊 **PPTX** (PowerPoint) | [presentation.pptx](presentation.pptx) |
| 📓 **Notebook** | [demo.ipynb](demo.ipynb) |

## 🧪 Running Tests

```bash
pytest tests/ -v --tb=short
```

All 31 tests should pass across the 4 validation phases:
- **Phase 1**: Forward model (phantom + projector)
- **Phase 2**: Reconstruction (LU, LSQR, FBP)
- **Phase 3**: Quality metrics (RMSE, PSNR, SSIM)
- **Phase 4**: Noise robustness & regularization

## 🛠 Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/scikit--image-F06030?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-image">
  <img src="https://img.shields.io/badge/pydicom-0A9EDC?style=flat" alt="pydicom">
  <img src="https://img.shields.io/badge/pytest-0A9EDC?style=flat&logo=pytest&logoColor=white" alt="pytest">
  <img src="https://img.shields.io/badge/Rich-FF6F00?style=flat" alt="Rich">
  <img src="https://img.shields.io/badge/python--pptx-2E77BC?style=flat" alt="python-pptx">
</p>

## 📚 References

- Shepp & Logan (1974). *The Fourier reconstruction of a head section.* IEEE Trans. Nucl. Sci.
- Paige & Saunders (1982). *LSQR: An algorithm for sparse linear equations and sparse least squares.* ACM Trans. Math. Softw.
- Golub & Van Loan (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
- Barre's DICOM Collection — [barre.dev/medical/samples](https://barre.dev/medical/samples/)

---

<div align="center">
  <sub>Built with ❤️ as a college project — CT Reconstruction using LU Decomposition</sub>
  <br>
  <sub>© 2026</sub>
</div>
