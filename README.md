# AIRL Internship Coding Assignment

This repository contains the solutions for the Vision Transformer (ViT) and Text-Driven Image Segmentation assignments.

## 📝 Table of Contents
* [Question 1: Vision Transformer on CIFAR-10](#question-1-vision-transformer-on-cifar-10)
  * [How to Run](#how-to-run-q1)
  * [Best Model Configuration](#best-model-configuration)
  * [Results](#results)
  * [Bonus: Analysis](#bonus-analysis)
* [Question 2: Text-Driven Image Segmentation with SAM 2](#question-2-text-driven-image-segmentation-with-sam-2)
  * [How to Run](#how-to-run-q2)
  * [Pipeline Description](#pipeline-description)
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
| Training Techniques | MixUp (alpha=0.8), Label Smoothing (0.1), Cosine LR Scheduler with Warmup (500 steps) |

### Results
The final classification accuracy on the CIFAR-10 test set is reported below.

| Metric                          | Score     |
| ------------------------------- | --------- |
| **Overall Test Accuracy (%)** | **78.54%**|

### (Bonus) Analysis
To maximize performance, I implemented several modern training techniques:
* **Strong Augmentation**: I used `RandAugment` in addition to standard crops and flips. This introduces significant diversity into the training data, which is crucial for helping Vision Transformers generalize well on smaller datasets like CIFAR-10 and preventing overfitting.
* **MixUp Regularization**: By linearly interpolating pairs of images and their labels, MixUp encourages the model to learn smoother decision boundaries. This provided a noticeable improvement over training without it.
* **AdamW and Cosine Scheduler**: Instead of a simple Adam optimizer or a step-based learning rate decay, I used AdamW combined with a cosine annealing scheduler with a warmup period. This combination is known to provide more stable training and better final convergence for Transformer models.

---

## Question 2: Text-Driven Image Segmentation with SAM 2

This section covers the implementation of text-prompted image segmentation using SAM 2, as detailed in `q2.ipynb`.

### How to Run (Q2)
1.  Open `q2.ipynb` in Google Colab.
2.  Ensure the runtime is set to use a GPU accelerator.
3.  Run all cells from top to bottom. The notebook will install dependencies, load models, accept a text prompt, and display the final segmented image.

### Pipeline Description
*(You need to fill this part in based on your Q2 notebook. Example below)*
1.  **Load Image & Prompt**: An image is loaded and the user provides a text prompt.
2.  **Generate Region Seeds**: I use Grounding DINO to interpret the text and generate a bounding box for the target object.
3.  **Segment with SAM 2**: This bounding box is passed to SAM 2 to produce a high-quality segmentation mask.
4.  **Display Mask**: The final mask is overlaid on the original image.

### Limitations
*(You need to fill this part in. Example below)*
* The pipeline's success is highly dependent on the initial bounding box from Grounding DINO.
* It struggles with highly ambiguous text prompts or heavily occluded objects.