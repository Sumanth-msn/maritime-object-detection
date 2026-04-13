# Maritime Object Detection using YOLOv8

## 📌 Overview

A real-time object detection system for maritime environments that identifies swimmers, boats, jetskis, buoys, and lifesaving equipment from aerial drone footage.

Built using YOLOv8, the system is optimized for high accuracy and real-world deployment scenarios such as search & rescue, surveillance, and beach safety.

---

## 🚀 Problem Statement

Manual monitoring of maritime environments is slow and error-prone.

This project solves:

* Detecting small objects (like swimmers) in large ocean scenes
* Real-time monitoring from drone footage
* Handling class imbalance and noisy data

---

## ⚙️ System Workflow

1. **Data Collection**

   * SeaDronesSee dataset (54K+ images)

2. **Preprocessing Pipeline**

   * COCO → YOLO format conversion
   * Image tiling (splitting large images)
   * Overlapping tiles to preserve boundary objects
   * Removal of corrupt/invalid labels

3. **Model Training**

   * YOLOv8-Large model
   * Trained for 20 epochs
   * Optimized using cosine learning rate scheduling and AdamW optimizer

4. **Evaluation**

   * Metrics: mAP, Precision, Recall

5. **Inference**

   * Supports image, batch, and real-time video detection

6. **Deployment**

   * Exported to ONNX / TensorRT for real-world usage

---

## 🧠 Key Features

* ✅ High Accuracy: 91.1% mAP@50
* ⚡ Real-time Inference (~45 ms/image)
* 🎯 Strong Small Object Detection (via tiling)
* 🧹 Clean Data Pipeline (label validation + preprocessing)
* 📦 Deployment Ready (multi-format export)

---

## 📊 Performance Metrics

* **Precision (91.2%)** → How many predicted objects are correct
* **Recall (89.5%)** → How many real objects were detected
* **mAP@50 (91.1%)** → Overall detection accuracy at 50% overlap
* **mAP@50–95 (58.7%)** → Stricter accuracy across multiple thresholds

---

## 🏗️ Model Architecture (Simplified)

* **Backbone** → Extracts features from image
* **Neck** → Combines features at multiple scales
* **Head** → Predicts object class and location

YOLOv8 uses an anchor-free detection approach, making it faster and simpler.

---

## ⚡ Key Improvements Made

* Image tiling → Improved small object detection
* Overlap tiling → Avoids cutting objects at edges
* Label cleaning → Prevents training noise
* Cosine learning rate → Stable training
* Mixed precision → Faster training

---

##  Results Summary

* Achieved **91.1% mAP@50**
* Real-time performance on GPU (~22 FPS)
* Balanced accuracy across all object classes

---

##  Applications

* Search & Rescue Operations
* Maritime Surveillance
* Beach Safety Monitoring
* Marine Research & Analytics

---

## 🛠️ Tech Stack

* Python
* YOLOv8 (Ultralytics)
* PyTorch
* OpenCV
* NumPy

---

## 📌 Key Takeaways

* Focused on real-world deployment, not just training accuracy
* Solved practical challenges like small objects and noisy labels
* Balanced speed vs accuracy for real-time systems

---

## 📎 Future Improvements

* Multi-camera tracking
* Edge deployment optimization
* Integration with alert systems

---

> Built as part of an internship project focusing on real-world AI deployment.

---

## Experimental Results

| Version | Resolution | Epochs | Best Epoch | mAP@50 | Precision | Recall | Status |
|-------|-----------|--------|-----------|-------|----------|-------|--------|
| v1.0 | 640×640 | 20 | 20 | **91.1%** | 91.2% | 89.5% | Production |
| v2.0 | 640×640 | 40 | 29 | 91.1% | 91.2% | 89.5% | Discontinued |
| v3.0 | 1024×1024 | 25 | 13 | 89.2% | 88.8% | 87.1% | Failed |

**v1.0 was selected as the final production model due to the optimal accuracy–speed tradeoff.**
