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
📂 Data Preparation
We evaluate our method on four datasets: RegDB, SYSU-MM01, LLCM, and VCM.
Please download the datasets and organize them in the data/ directory as follows:

text
data/
├── RegDB/
├── SYSU-MM01/
├── LLCM/
└── VCM/

## 📊 Results

### Comparison with State-of-the-Art Methods

We compare CSPL-ReID with existing lifelong VI-ReID methods. Results are reported as mAP / Rank-1 (%). 
* **Reference**: JointTrain (upper bound) and SFT (sequential fine-tuning without anti-forgetting).
* **Replay**: Methods that retain or reuse images from previous tasks.
* **Non-Replay**: Methods that do not store original images from previous tasks (Our setting).

| Method | Type | RegDB | SYSU | LLCM | VCM | AVG |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| JointTrain | Reference | 73.6 / 78.5 | 57.2 / 61.2 | 50.4 / 42.5 | 21.8 / 37.0 | 50.8 / 54.8 |
| SFT | Reference | 5.2 / 3.9 | 18.8 / 18.8 | 13.4 / 8.4 | 15.4 / 26.6 | 13.2 / 14.4 |
| LwF | Replay | 26.9 / 26.1 | 30.9 / 30.3 | 39.1 / 34.5 | 18.2 / 30.0 | 28.8 / 30.2 |
| iCaRL | Replay | 42.4 / 41.3 | 32.4 / 32.4 | 41.9 / 36.8 | 18.5 / 30.7 | 33.8 / 35.3 |
| BiC | Replay | 40.9 / 39.3 | 22.8 / 21.9 | 43.3 / 37.5 | 16.4 / 27.4 | 30.9 / 31.5 |
| WA | Replay | 44.9 / 43.9 | 34.6 / 34.4 | 44.1 / 38.7 | 18.7 / 29.3 | 35.6 / 36.6 |
| PTKP | Replay | 50.3 / 47.9 | 26.6 / 25.8 | 43.2 / 36.5 | 17.9 / 27.4 | 34.5 / 34.4 |
| TTQK | Replay | 60.3 / 58.3 | 33.6 / 30.9 | 45.8 / 39.6 | 19.4 / 32.2 | 39.8 / 40.2 |
| PP-IPG | Replay | 59.0 / 58.8 | 32.7 / 35.2 | 44.4 / 50.8 | 35.3 / 23.3 | 42.8 / 42.0 |
| LSTKC | Non-Replay | 35.0 / 35.4 | 32.7 / 34.3 | 39.6 / 35.1 | 15.1 / 25.3 | 30.6 / 32.5 |
| DKP | Non-Replay | 30.3 / 32.7 | 30.9 / 32.8 | 37.3 / 32.2 | 15.2 / 26.3 | 28.4 / 31.0 |
| DASK | Non-Replay | 21.6 / 25.6 | 28.1 / 32.3 | 39.6 / 36.2 | 14.1 / 27.4 | 25.9 / 30.4 |
| LSTKC++ | Non-Replay | 19.2 / 19.0 | 27.1 / 30.3 | 32.9 / 29.2 | 16.1 / 29.2 | 23.8 / 26.9 |
| CKDA | Non-Replay | 57.0 / 60.4 | 34.7 / 37.3 | 37.8 / 31.5 | 15.6 / 28.2 | 36.3 / 39.4 |
| **Ours** | **Non-Replay** | **64.5 / 69.5** | **38.4 / 41.2** | **39.2 / 33.3** | **16.8 / 29.2** | **39.7 / 43.3** |
