# Edge AI Face Recognition Optimization via INT8 Post-Training Quantization

This repository contains the code and engineering pipeline for the MSc Artificial Intelligence thesis:
**"Optimizing Deep Face Recognition Models for Edge AI Computer Vision Deployments"**.

## Project Overview
The primary goal is designing, implementing, and evaluating an end-to-end edge deployment pipeline for face recognition. The project focuses on model compression and latency reduction of state-of-the-art architectures (**ArcFace**) via **INT8 Post-Training Quantization (PTQ)**.

The objective is to drastically minimize memory footprint and inference latency, enabling real-time on-device inference on resource-constrained hardware (embedded systems, mobile, edge devices) while preserving recognition accuracy.

## Technical Architecture & Pipeline
- **Base Architecture:** ArcFace backbone via the DeepFace framework.
- **Feature Extraction:** High-dimensional 512-D identity embeddings.
- **Optimization Strategy:** Post-Training Quantization (PTQ) to INT8 precision using `tf.lite.TFLiteConverter`.
- **Target Runtime:** TensorFlow Lite (TFLite) for lightweight edge execution.
- **Evaluation Metrics:** Cosine Similarity retention across Intra-Class and Inter-Class verification pairs.

## Verification & Dataset
- **Verification Samples:** Aligned 112x112 face images evaluated under controlled intra-class (same identity) and inter-class (different identity) conditions.
- **Pretrained Weights:** Large-scale pretrained ArcFace weights utilized for robust feature representations prior to quantization.

## Key Outcomes
- **Model Compression:** Substantial memory footprint reduction (~4x reduction moving from FP32 to INT8).
- **Latency Gain:** Accelerated on-device inference latency suitable for edge compute.
- **Accuracy Retention:** Maintained strong cosine similarity separation between matching and non-matching identity pairs post-quantization.
