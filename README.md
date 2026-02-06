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
pip install torch numpy matplotlib scipy
```

## Quick Start
```bash
# Run cantilever beam example (Section 5.1)
python beam_straight_onekan_triangle.py
```

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
