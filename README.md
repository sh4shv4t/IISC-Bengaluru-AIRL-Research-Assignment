# AIRL Internship Coding Assignment Submission

This repository contains my solutions for the Vision Transformer (ViT) and Text-Driven Image Segmentation assignments for the AIRL Research Internship.

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

This section covers the implementation of a Vision Transformer for image classification on the CIFAR-10 dataset, as detailed in `q1.ipynb`.

### How to Run (Q1)
1.  Open `q1.ipynb` in Google Colab.
2.  Ensure the runtime is set to use a GPU accelerator.
3.  Run all cells from top to bottom to install dependencies, load data, train the model, and evaluate its final performance.

### Best Model Configuration
The configuration below achieved the final test accuracy.

| Hyperparameter      | Value                |
| ------------------- | -------------------- |
| Patch Size          | 4x4 |
| Embedding Dimension | 192 |
| Number of Heads (MHSA)| 8 |
| Number of Layers    | 8 |
| MLP Hidden Dimension| 768 |
| Optimizer           | AdamW |
| Learning Rate       | 3e-4 |
| Weight Decay        | 0.05 |
| Batch Size          | 128 |
| Training Epochs     | 50 |
| Augmentations       | RandomCrop, RandomHorizontalFlip, RandAugment |
| Training Techniques | MixUp (alpha=0.8), Label Smoothing (0.1), Cosine LR Scheduler with Warmup |

### Results
The final classification accuracy on the CIFAR-10 test set is reported below.

| Metric                          | Score     |
| ------------------------------- | --------- |
| **Overall Test Accuracy (%)** | **78.54%**|

### (Bonus) Analysis
To maximize performance on CIFAR-10, I implemented several modern training techniques that are crucial for training Vision Transformers effectively on smaller datasets:
* **Strong Augmentation**: I used `RandAugment` in addition to standard crops and flips. This introduces significant diversity into the training data, which acts as a strong regularizer and is essential for helping ViTs generalize well.
* **MixUp & Label Smoothing**: By linearly interpolating pairs of images and their labels (MixUp) and softening the target labels (Label Smoothing), the model is encouraged to learn smoother decision boundaries and reduce overconfidence, leading to better calibration and accuracy.
* **AdamW and Cosine Scheduler**: Instead of a simple Adam optimizer, I used AdamW combined with a cosine annealing scheduler with a warmup period. This combination provides more stable training and better final convergence for Transformer architectures.

---

## Question 2: Text-Driven Image & Video Segmentation

This section covers the implementation of text-prompted segmentation on a single image and the bonus video extension, as detailed in `q2.ipynb`.

### How to Run (Q2)
1.  Open `q2.ipynb` in Google Colab.
2.  Ensure the runtime is set to use a GPU accelerator.
3.  Run all cells from top to bottom. The notebook will download a sample video, prompt the user for a text input, process the video frame-by-frame, and save the result as `output_video_final.mp4`.

### Pipeline Description
My approach for text-prompted segmentation uses a two-stage pipeline combining a text-to-box model with a promptable segmentation model.
1.  **Load Media**: An image or video frame is loaded.
2.  **Accept Text Prompt**: The user inputs a text string describing the object of interest (e.g., "dog").
3.  **Generate Region Seeds**: I use the **`IDEA-Research/grounding-dino-base`** model to interpret the text prompt and generate bounding box coordinates (region seeds) for the target object.
4.  **Segment with SAM**: These bounding box seeds are fed into the **`facebook/sam-vit-base`** model.
5.  **Display Mask**: SAM outputs a high-quality segmentation mask, which is then overlaid on the original image or frame.

### (Bonus) Video Extension
I extended the pipeline to perform text-driven video object segmentation. The approach is as follows:
1.  **Initial Detection**: Grounding DINO runs on the initial frames of the video to find the object specified by the text prompt.
2.  **Tracking-by-Detection**: Once an initial bounding box is found, it is saved. This **same static box** is then used as the prompt for SAM on all subsequent frames. This allows for tracking the object as long as it stays within the initial detection area.
3.  **Mask Generation & Output**: SAM generates a mask for each frame, and the masked frames are compiled into a final output video.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RrZmArab08j2ua7DbFp0PCPpwf_JaDNY?usp=sharing)


### Limitations
The primary limitation of this pipeline is its simple approach to tracking:
* **Static Bounding Box**: The core weakness is that the bounding box from the initial detection is re-used for all subsequent frames. As a result, the mask accuracy degrades significantly when the object undergoes **fast motion** or moves outside its initial detection area.
* **Detection Dependency**: The entire process is contingent on a successful initial detection by Grounding DINO. If the object is not found in the first few frames, the tracking fails.
* **Ambiguity**: Ambiguous prompts (e.g., "the person") could cause the initial detection to lock onto the wrong object if multiple instances are present. A more advanced solution would incorporate mask propagation or a more sophisticated tracking algorithm to update the prompt for SAM on a frame-by-frame basis.