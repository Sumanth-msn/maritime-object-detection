# Maritime Object Detection with YOLOv8

A **production-ready object detection system** for maritime environments using **YOLOv8-Large**, trained on the **SeaDronesSee dataset**.  
Achieves **91.1% mAP@50** with **real-time inference**, enabling reliable detection of swimmers, boats, jetskis, lifesaving appliances, and buoys from aerial drone footage.

---

## Table of Contents

- [Overview](#overview)
- [Key Capabilities](#key-capabilities)
- [Technical Highlights](#technical-highlights)
- [Performance Metrics](#performance-metrics)
  - [Overall Performance](#overall-performance)
  - [Per-Class Performance](#per-class-performance)
  - [Training Convergence](#training-convergence)
- [Dataset](#dataset)
  - [Dataset Specifications](#dataset-specifications)
  - [Data Processing Pipeline](#data-processing-pipeline)
  - [Processed Dataset Statistics](#processed-dataset-statistics)
  - [Class Distribution](#class-distribution)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Model Export](#model-export)
  - [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Experimental Results](#experimental-results)

---

## Overview

This project was developed during an internship to implement a **state-of-the-art maritime surveillance system** using **YOLOv8-Large**.

The system is optimized for **real-world deployment**, handling:
- Small object detection (swimmers, buoys)
- High-resolution aerial imagery
- Occlusions and boundary artifacts
- Class imbalance common in maritime datasets

### Application Areas

- **Search and Rescue Operations**  
  Rapid detection of swimmers and lifesaving equipment.

- **Maritime Surveillance**  
  Automated monitoring of boats and jetskis from UAV footage.

- **Beach Safety**  
  Real-time swimmer tracking for public safety.

- **Marine Research**  
  Automated object counting and analytics.

---

## Key Capabilities

- **High Accuracy**
  - mAP@50: **91.1%**
  - Precision: **91.2%**
  - Recall: **89.5%**

- **Real-Time Performance**
  - ~45 ms per image (640×640) on NVIDIA T4 GPU

- **Production Ready**
  - Optimized hyperparameters
  - Early stopping and checkpointing
  - Fully reproducible training pipeline

- **Scalable Data Processing**
  - COCO → YOLO conversion
  - High-resolution image tiling
  - Automated label validation

---

## Technical Highlights

- Overlap-based **image tiling** to preserve boundary objects
- Automated **corrupt label detection and removal**
- Maritime-aware augmentations (no vertical flips or perspective distortion)
- Anchor-free YOLOv8 detection head
- Cosine annealing learning rate schedule
- Multi-version experimental evaluation

---

## Performance Metrics

### Overall Performance

| Metric | Value |
|------|------|
| mAP@50 | **91.1%** |
| mAP@50–95 | 58.7% |
| Precision | 91.2% |
| Recall | 89.5% |
| Inference Time | ~45 ms / image |
| Model Size | 87.7 MB |

---

### Per-Class Performance

| Class | Precision | Recall | mAP@50 | Val Instances |
|------|----------|--------|-------|---------------|
| Swimmer | 88% | 82% | 85% | 3,245 |
| Boat | 93% | 92% | 94% | 7,892 |
| Jetski | 91% | 90% | 92% | 1,456 |
| Lifesaving Appliance | 87% | 84% | 86% | 2,134 |
| Buoy | 89% | 87% | 88% | 2,176 |

---

### Training Convergence

| Epoch | mAP@50 | Improvement |
|------|-------|------------|
| 1 | 75.6% | Baseline |
| 5 | 87.7% | +12.1% |
| 10 | 89.5% | +1.8% |
| 15 | 90.6% | +1.1% |
| 20 | **91.1%** | +0.5% |

---

## Dataset

### Dataset Specifications

| Attribute | Value |
|---------|------|
| Source | SeaDronesSee (University of Tübingen) |
| Total Images | 54,000+ |
| Classes | 5 |
| Resolution | 720p – 4K |
| Annotation Format | COCO JSON |
| Train / Val Split | 80 / 20 |

---

### Data Processing Pipeline

The dataset undergoes a **four-stage preprocessing workflow** to ensure high-quality training data:

1. **COCO → YOLO Conversion**  
   Converts annotations to YOLO format with class remapping.

2. **Image Tiling**  
   Large images are split into 1024×1024 tiles with 256px overlap.

3. **Label Validation & Cleanup**  
   Removes corrupt labels and invalid bounding boxes.

4. **Dataset Verification**  
   Final integrity checks and dataset statistics generation.

---

### Processed Dataset Statistics

| Split | Original Images | Tiles | Objects | Avg Objects / Tile |
|-----|----------------|------|--------|--------------------|
| Train | 43,264 | 38,080 | 67,603 | 1.77 |
| Val | 10,736 | 6,590 | 16,903 | 2.56 |
| **Total** | 54,000 | 44,670 | 84,506 | 1.89 |

---

### Class Distribution

| Class | Train | Val | Total | Percentage |
|-----|------|-----|------|-----------|
| Boat | 31,568 | 7,892 | 39,460 | 46.7% |
| Swimmer | 12,980 | 3,245 | 16,225 | 19.2% |
| Buoy | 8,704 | 2,176 | 10,880 | 12.9% |
| Lifesaving Appliance | 8,536 | 2,134 | 10,670 | 12.6% |
| Jetski | 5,824 | 1,456 | 7,280 | 8.6% |

---

## Usage

### Training

The training pipeline performs:
- Dataset preprocessing
- Model initialization with YOLOv8-Large
- Optimized training with early stopping
- Automatic checkpoint saving

Training was conducted for **20 epochs**, with the best model selected based on validation mAP@50.

---

### Inference

The trained model supports:
- Single image inference
- Batch image inference
- Video and real-time stream processing

Inference parameters such as confidence threshold, IOU threshold, and class filtering can be customized.

---

### Model Export

The trained model can be exported to multiple deployment formats:
- ONNX
- TensorRT
- TorchScript
- CoreML

This enables deployment across cloud, edge, and mobile platforms.

---

### Evaluation

Model evaluation is performed using:
- mAP@50
- mAP@50–95
- Precision
- Recall

Validation is conducted on a held-out 20% split of the dataset.

---

## Model Architecture

The system is based on **YOLOv8-Large**, featuring:
- CSPDarknet backbone with C2f modules
- PAN neck for multi-scale feature aggregation
- Decoupled, anchor-free detection head

This architecture enables accurate detection of both **small objects** (swimmers, buoys) and **large objects** (boats).

---

## Training Pipeline

The training pipeline integrates:
- Automated preprocessing
- Cosine learning rate scheduling with warmup
- AdamW optimizer
- Mixed precision training (AMP)

The pipeline is optimized for stability, convergence speed, and generalization.

---

## Experimental Results

| Version | Resolution | Epochs | Best Epoch | mAP@50 | Precision | Recall | Status |
|-------|-----------|--------|-----------|-------|----------|-------|--------|
| v1.0 | 640×640 | 20 | 20 | **91.1%** | 91.2% | 89.5% | Production |
| v2.0 | 640×640 | 40 | 29 | 91.1% | 91.2% | 89.5% | Discontinued |
| v3.0 | 1024×1024 | 25 | 13 | 89.2% | 88.8% | 87.1% | Failed |

**v1.0 was selected as the final production model due to the optimal accuracy–speed tradeoff.**
