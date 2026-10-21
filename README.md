# CSPL-ReID: Cross-modal Structure-Preserved Lifelong Learning for Visible-Infrared Person Re-identification

This is the official PyTorch implementation of the paper: **CSPL-ReID: Cross-modal Structure-Preserved Lifelong Learning for Visible-Infrared Person Re-identification**.

## 📖 Introduction

Visible-infrared person re-identification (VI-ReID) enables around-the-clock person retrieval. In a lifelong-learning setting, a model must adapt to a sequence of cross-modal tasks while the original images from previous tasks are unavailable. 

To address the structural drift during sequential updates, we propose **CSPL-ReID**. It contains three core components:
- **CSAD**: Aligns RGB/IR class centers and preserves within-batch relations.
- **PDRC**: Uses a frozen previous model and prototype bank to retain historical relations.
- **GSPC**: Limits the shift toward the final task via group-specific parameter fusion.

## 🛠️ Requirements

- Python 3.8
- PyTorch 2.0.0
- CUDA 11.8

You can install the required packages via:
```bash
pip install -r requirements.txt# CSPL-ReID
