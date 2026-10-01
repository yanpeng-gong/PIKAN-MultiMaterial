# PIKAN-MultiMaterial

Physics-Informed Kolmogorov-Arnold Networks for multi-material elasticity problems in electronic packaging.

[![Paper](https://img.shields.io/badge/Paper-Applied%20Mathematical%20Modelling-blue)](https://doi.org/10.1016/j.apm.2026.116793)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

## Overview

Official implementation of **PIKAN** - a method that uses Physics-Informed Kolmogorov-Arnold Networks to solve multi-material elasticity problems without domain decomposition.

**Key Features:**
- Single network for entire multi-material domain
- No interface continuity constraints needed
- B-splines naturally handle material discontinuities
- Energy-based formulation (Deep Energy Method)

## Installation
```bash
git clone https://github.com/yanpeng-gong/PIKAN-MultiMaterial.git
cd PIKAN-MultiMaterial
```

Create and activate a virtual environment:

```powershell
# Windows PowerShell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

For an NVIDIA GPU on Windows or Linux, install the CUDA 13.0 build of PyTorch first, then the other dependencies:

```bash
python -m pip install -r requirements-gpu-cu130.txt
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The final command should print a version ending in `+cu130`, `13.0`, and `True`. PyTorch's [official installation page](https://pytorch.org/get-started/locally/) lists other CUDA builds if your driver needs a different version.

For a CPU-only environment, install the standard dependencies instead:

```bash
python -m pip install -r requirements.txt
```

The example currently requires CUDA. It creates its output directories automatically under the repository.

## Quick Start
```bash
# Run cantilever beam example (Section 5.1)
python beam_straight_onekan_triangle.py
```

To export results from an existing checkpoint without training again:

```bash
python beam_straight_onekan_triangle.py --checkpoint Net/model_k_triangle_grid10order320261001_175915.pth
```

Replace the checkpoint name with your saved `.pth` file. Models are saved in `Net/`. CSV data and loss history are written to `results/`, with plots in `results/plots/`.
Loss history is available only for a new training run; a `.pth` checkpoint stores model weights but not past loss values.

The Abaqus comparison plots need `data/abaqus/coord.xlsx`, `u1all.xlsx`, `u2all.xlsx`, and `uall.xlsx`. FEM point predictions need `data/absoluteerror/coord.xlsx`; absolute-error plots additionally need `absoluteerror_ux.xlsx` and `absoluteerror_uy.xlsx` in that directory. Missing input files are listed at runtime, and the corresponding optional plots are skipped. The program creates the input directories but cannot generate the external FEM data.

## Citation
```bibtex
@article{Gong2026PIKAN,
  author  = {Gong, Yanpeng and He, Yida and Mei, Yue and Qin, Fei and 
             Zhuang, Xiaoying and Rabczuk, Timon},
  title   = {Physics-informed {Kolmogorov-Arnold} networks for multi-material 
             elasticity problems in electronic packaging},
  journal = {Applied Mathematical Modelling},
  volume  = {156},
  pages   = {116793},
  year    = {2026},
  doi     = {10.1016/j.apm.2026.116793}
}
```

**Paper:** Gong et al., Applied Mathematical Modelling, 2026. [Link](https://doi.org/10.1016/j.apm.2026.116793)

## Contact

**Yanpeng Gong**  
Beijing University of Technology  
Email: yanpenggong@gmail.com

For questions or collaborations, please open an issue or contact via email.
