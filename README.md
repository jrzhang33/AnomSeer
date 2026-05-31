<h2 align="center">
AnomSeer: Reinforcing Multimodal LLMs to Reason for Time-Series Anomaly Detection
<br><br>
<img src="./assets/logo.jpg" width="300" alt="AnomSeer Logo">
<br>
<b>@ <a href="https://icml.cc/">ICML 2026</a></b>
</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2602.08868">
    <img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square&logo=arxiv" alt="arXiv Paper"></a>
  &nbsp;
  <a href="https://github.com/jrzhang33/AnomSeer">
    <img src="https://img.shields.io/badge/GitHub-Project-181717?style=flat-square&logo=github" alt="GitHub Project"></a>
</p>

<p align="center">
  <img src="./assets/case-1.png" width="50%" alt="AnomSeer Overview">
</p>

**AnomSeer** is a reinforcement-learning post-training framework for time-series anomaly detection (TSAD) with multimodal LLMs. It unifies anomaly **classification**, **localization**, and **explanation** by grounding MLLM reasoning in fine-grained, classical TSAD evidence through two components:

- **ExpCoT** — expert chain-of-thought traces generated from classical statistical methods (FFT, Matrix Profile, gradient analysis) that provide structured, quantitatively verifiable supervision.
- **TimerPO** — a novel RL algorithm that uses Optimal Transport to measure semantic alignment between model reasoning and ExpCoT, then applies orthogonal projection to inject this signal without interfering with the primary detection objective.

With Qwen2.5-VL-3B/7B-Instruct as backbone, AnomSeer outperforms GPT-4o and Gemini-2.5-Pro on anomaly classification and localization across AnomLLM, VisualTimeAnomaly, and TSB-UAD benchmarks.

---

# Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
  - [AnomLLM (Training Data)](#1-anomllm-training-data)
  - [VisualTimeAnomaly](#2-visualtimeanomaly-evaluation)
  - [TSB-UAD](#3-tsb-uad-evaluation)
  - [Preprocess into Parquet](#4-preprocess-into-parquet)
- [Training](#training)
- [Evaluation](#evaluation)
- [Key Design Choices](#key-design-choices)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

# Installation

```bash
conda create -n anomseer python=3.11 -y
conda activate anomseer

pip3 install -r requirements.txt
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .

pip3 install -r requirements_anomseer.txt
pip3 install git+https://github.com/ahstat/affiliation-metrics-py.git
```

> Requires CUDA 12.4 (for `torch 2.6.0` and `flash-attn`). `flash-attn` must be installed separately with `--no-build-isolation`.

---

# Data Preparation

AnomSeer is **trained** on AnomLLM (synthetic) and **evaluated** on AnomLLM, VisualTimeAnomaly, and TSB-UAD.

## 1. AnomLLM (Training Data)

Clone [AnomLLM](https://github.com/Rose-STL-Lab/AnomLLM), set up its environment, and generate the synthetic dataset:

```bash
git clone https://github.com/Rose-STL-Lab/AnomLLM.git
cd AnomLLM
conda env create --file environment.yml && conda activate anomllm && poetry install --no-root
bash synthesize.sh
```

## 2. VisualTimeAnomaly (Evaluation)

Download from: [https://github.com/mims-harvard/VisualTimeAnomaly](https://github.com/mims-harvard/VisualTimeAnomaly)

## 3. TSB-UAD (Evaluation)

Download from: [https://github.com/decisionintelligence/TAB](https://github.com/decisionintelligence/TAB)

## 4. Preprocess into Parquet

```bash
python multimodal_data_processing/anom.py \
    --root_dir /path/to/AnomLLM/data/synthetic \
    --out_dir ./data/anol_processed_mllm_data
```

This produces `train_full.parquet` (3,200 samples) and `test_full.parquet` under `./data/anol_processed_mllm_data/`.

---

# Training

Edit `MODEL_PATH` in `example/timerpo_trainer/run_anomseer.sh` to select the backbone:

```bash
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct   # 3B variant
# MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # 7B variant
```

Then launch training (requires 4 GPUs by default):

```bash
bash example/timerpo_trainer/run_anomseer.sh
```

Checkpoints are saved to `./checkpoints/`.

To override the data paths without editing the script:

```bash
TRAIN_FILE=./data/anol_processed_mllm_data/train_full.parquet \
VAL_FILE=./data/anol_processed_mllm_data/test_full.parquet \
bash example/timerpo_trainer/run_anomseer.sh
```

---

# Evaluation

Set `EVAL=True` inside `example/timerpo_trainer/run_anomseer.sh` and point `VAL_FILE` to the desired benchmark:

```bash
EVAL=True
VAL_FILE=./data/anol_processed_mllm_data/test_full.parquet
```

Then run:

```bash
bash example/timerpo_trainer/run_anomseer.sh
```

Metrics reported in the logs:
- **Affinity-Precision / Affinity-Recall / Affinity-F1** — temporal localisation quality
- **Classification Accuracy** — anomaly type identification

To evaluate on VisualTimeAnomaly or TSB-UAD, preprocess those datasets with `multimodal_data_processing/anom.py` into the same parquet format and set `VAL_FILE` accordingly.


---

# Citation

```bibtex
@inproceedings{zhang2026anomseer,
  title     = {AnomSeer: Reinforcing Multimodal LLMs to Reason for Time-Series Anomaly Detection},
  author    = {Zhang, Junru and Feng, Lang and Shi, Haoran and Guo, Xu and Yu, Han and Dong, Yabo and Xu, Duanqing},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

# Acknowledgements

We thank the [veRL](https://github.com/volcengine/verl) project for foundational RL infrastructure and [AnomLLM](https://github.com/Rose-STL-Lab/AnomLLM) for the synthetic TSAD benchmark and data generation code.
