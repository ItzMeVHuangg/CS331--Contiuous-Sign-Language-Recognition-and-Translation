# Continuous Sign Language Recognition and Translation (CSLR & SLT) Framework

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License" /></a>
  <img src="https://img.shields.io/badge/Python-3.8+-3c873a?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Models-ResNet%20%7C%20Transformer-blue?style=flat-square" alt="Model" />
  <img src="https://img.shields.io/badge/Dataset-PHOENIX--2014T-orange?style=flat-square" alt="Dataset" />
</p>

This repository presents an end-to-end framework for Continuous Sign Language Recognition (CSLR) and Sign Language Translation (SLT), rigorously evaluated on the **PHOENIX 2014T** dataset.

![Main System Architecture](./main_system.jpg)

## 1. System Pipeline Workflow
The framework is designed based on a **Sign-to-Gloss-to-Text** architecture, conceptually divided into two primary stages. The multimodal data flows through the following computational modules:

* **Input:** Raw continuous sign language videos.

![System Pipeline Architecture](./pipeline.png)

### STAGE 1: Continuous Sign Language Recognition (CSLR)
- **Preprocessing Module:** Handles spatial normalization, center cropping, and temporal sampling of the input video frames.
- **Spatial Feature Extractor:** Responsible for capturing fine-grained morphological and spatial representations from individual frames or skeletal keypoints. This module is designed to support comprehensive ablation studies across 5 distinct vision backbones:
  - `ResNet18-2D` (Lightweight 2D CNN)
  - `ResNet34` (Deep 2D CNN)
  - `ResNet18-3D` (Spatio-temporal 3D CNN)
  - `Video Swin` (3D Video Transformer)
  - `Mediapipe` (Skeleton-based landmark extraction)
- **Temporal Modeler:** Captures long-range contextual dependencies and spatio-temporal motion trajectories. This module alternately employs 3 temporal backbones:
  - `BiLSTM` (Bidirectional Recurrent Neural Network)
  - `Transformer` (Self-Attention mechanism)
  - `Conformer` (CNN-Transformer hybrid architecture)
- **CSLR Head (Recognition & Decoding):** Utilizes Connectionist Temporal Classification (**CTC**) Loss to optimize for weakly-aligned sequence data. It integrates 2 decoding algorithms to emit the intermediate Gloss sequence:
  - `Beam Search Decoding` (Global probability optimization)
  - `Greedy Decoding` (Local, high-throughput decoding)
- **Late Fusion Protocol:** Employs a Cross-Attention mechanism to fuse two distinct information modalities: Visual features extracted from the Temporal Modeler and semantic Gloss Embeddings.

### STAGE 2: Sign Language Translation (SLT)
- **SLT Backbone (Translation Module):** Leverages an autoregressive **Seq2Seq Transformer** (standard Encoder-Decoder architecture) to map the fused multimodal representations into natural language sequences.
- **Output:** Fluent and grammatically complete translated sentences in the target language (German).

---

## 2. Dataset Overview
To empirically validate the robustness and practical efficacy of the proposed architecture, the system is trained and evaluated on the **RWTH-PHOENIX-Weather 2014T** corpus. Within the AI and Computer Vision research community, this dataset serves as a stringent benchmark standard specifically curated for multi-task sign language modeling (joint CSLR and SLT).

![Evaluation Metrics & Results](./evaluation.jpg)

Extracted directly from broadcast weather forecasts on German public television, PHOENIX-2014T offers an exceptionally high degree of linguistic realism, featuring performances by 9 professional deaf interpreters. Surpassing the limitations of studio-simulated data, PHOENIX-2014T provides a large-scale, highly standardized parallel corpus.

Specifically, the dataset challenges the computational model with:
- A massive scale of **8,257 continuous sign language sentences**, captured at a frame rate of 25 FPS.
- A profound semantic discrepancy (domain shift): the model must learn a highly non-linear mapping from a source sign vocabulary (Gloss space) of **1,099 words** to a significantly broader target language (German) vocabulary space encompassing **2,887 words**.

This complex linguistic structure and massive statistical scale make PHOENIX-2014T the ideal testbed to measure the representational limits and generalization capabilities of deep neural architectures such as ResNet, Video Swin, and Transformers.

---

## 3. Team Members

| Member | ID |
| :--- | :--- |
| Truong Hoang Thanh An | 23520032 |
| Nguyen Xuan An | 23520023 |
| Vu Viet Hoang | 23520548 |
| Mai Thai Binh | 23520158 |
