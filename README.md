# AIRL Research Internship Assessment

This repository contains my submission for the AIRL research internship assessment. It features a from-scratch implementation of a Vision Transformer (ViT) on CIFAR-10 and a text-driven video segmentation pipeline using Grounding DINO and the Segment Anything Model (SAM).

## 📝 Table of Contents
* [Question 1: Vision Transformer on CIFAR-10](#question-1-vision-transformer-on-cifar-10)
  * [How to Run (Q1)](#how-to-run-q1)
  * [Best Model Configuration](#best-model-configuration)
  * [Results](#results)
  * [Bonus: Analysis](#bonus-analysis)
* [Question 2: Text-Driven Image & Video Segmentation](#question-2-text-driven-image--video-segmentation)
  * [How to Run (Q2)](#how-to-run-q2)
  * [Pipeline Description](#pipeline-description)
  * [Bonus: Video Extension](#bonus-video-extension)
  * [Limitations](#limitations)

---

## Question 1: Vision Transformer on CIFAR-10

This notebook (`q1.ipynb`) contains a from-scratch implementation of the Vision Transformer architecture, trained and evaluated on the CIFAR-10 dataset with the objective of maximizing test accuracy.

### How to Run (Q1)
1.  Open `q1.ipynb` in Google Colab.
2.  Select a GPU-accelerated runtime (`T4` or similar).
3.  Execute all cells sequentially from top to bottom.

### Best Model Configuration
The final reported accuracy was achieved using the following configuration:

| Hyperparameter      | Value                |
| ------------------- | -------------------- |
| Patch Size          | 4x4 |
| Embedding Dimension | 192 |
| Transformer Layers (Depth) | 8 |
| MHSA Heads | 8 |
| MLP Hidden Dimension| 768 |
| Optimizer           | AdamW |
| Learning Rate       | 3e-4 |
| Weight Decay        | 0.05 |
| Batch Size          | 128 |
| Training Epochs     | 50 |
| Augmentations       | RandomCrop, RandomHorizontalFlip, RandAugment |
| Regularization      | MixUp (α=0.8), Label Smoothing (ε=0.1), Cosine LR Scheduler with Warmup |

### Results
| Metric                          | Score     |
| ------------------------------- | --------- |
| **Overall Test Accuracy (%)** | **78.54%**|

### (Bonus) Analysis
My strategy focused on implementing modern best practices for training ViTs, which are notoriously data-hungry, on a small-scale dataset like CIFAR-10. This resulted in an **11.22% accuracy improvement** over a baseline configuration, from 67.32% to 78.54%.

* **Architectural Scaling**: I methodically increased the model's capacity by expanding the embedding dimension (128 → 192) and adding more Transformer layers (6 → 8). This allowed the network to capture more complex inter-patch relationships.
* **Advanced Regularization**: To combat overfitting from the increased model capacity, I employed a suite of strong regularization techniques:
    * **Data Augmentation**: `RandAugment` was critical for artificially expanding the dataset's diversity.
    * **MixUp & Label Smoothing**: These techniques prevent the model from becoming overconfident in its predictions by encouraging smoother decision boundaries.
* **Optimizer & Schedule**: I used the `AdamW` optimizer, which decouples weight decay from the gradient update, along with a **Cosine Annealing scheduler** and a warmup phase. This combination is highly effective for stabilizing Transformer training and finding better optima.
* **Sufficient Training**: The number of training epochs was increased (20 → 50) to ensure the model had sufficient iterations to converge properly with the given learning rate schedule.

---

## Question 2: Text-Driven Image & Video Segmentation

This notebook (`q2.ipynb`) implements a pipeline that uses a natural language prompt to perform segmentation on a single image and extends this capability to video.

### How to Run (Q2)
1.  Open `q2.ipynb` in Google Colab.
2.  Select a GPU-accelerated runtime.
3.  Run all cells sequentially. The notebook will download a sample video, prompt for a text input, process the video, and save the result as `output_video_final.mp4`.

### Pipeline Description
The pipeline uses a two-stage, prompt-based approach:
1.  **Text-to-Box Generation**: The user provides a text prompt (e.g., "dog"). This prompt is fed to the **`IDEA-Research/grounding-dino-base`** model, which identifies and generates bounding box coordinates for the corresponding object in the input image or frame.
2.  **Box-to-Mask Generation**: The generated bounding boxes are then used as prompts for the **`facebook/sam-vit-base`** model. SAM uses these boxes as spatial cues to produce a precise, high-quality segmentation mask for the target object.

### (Bonus) Video Extension
The pipeline was successfully extended to perform text-driven video object segmentation.
1.  **Initial Detection**: Grounding DINO is run on the first few frames of the video until the object specified by the text prompt is successfully detected.
2.  **Tracking-by-Detection**: The initial bounding box is then held static and used as the sole prompt for SAM on all subsequent frames of the video clip.
3.  **Mask Generation & Output**: SAM generates a mask for each frame based on this initial box, and the masked frames are compiled into a final output video.

For the bonus part of the question- 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RrZmArab08j2ua7DbFp0PCPpwf_JaDNY?usp=sharing)


### Limitations
The primary limitation lies in the simple tracking-by-detection strategy:
* **Static Bounding Box**: The core weakness is that the initial bounding box is not updated frame-to-frame. As I observed, this causes the mask to become **inaccurate during fast object motion** or when the object moves significantly from its initial position.
* **Detection Dependency**: The entire process is contingent on a successful initial detection by Grounding DINO. If the object isn't found in the first few frames, the tracking cannot start.
* **Ambiguity**: Ambiguous prompts could cause the initial detection to lock onto the wrong object. A more advanced solution would incorporate a dedicated tracking algorithm or use mask propagation from frame t-1 as an additional prompt for frame t.