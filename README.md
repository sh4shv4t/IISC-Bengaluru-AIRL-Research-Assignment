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
  * [Bonus: Video Extension](#bonus-video-extension)

---

## Question 1: Vision Transformer on CIFAR-10

[cite_start]This section covers the implementation of a Vision Transformer for image classification on the CIFAR-10 dataset, as detailed in `q1.ipynb`. [cite: 10, 11]

### How to Run (Q1)
1.  [cite_start]Open `q1.ipynb` in Google Colab. [cite: 3, 29]
2.  Ensure the runtime is set to use a GPU accelerator.
3.  Run all cells from top to bottom to install dependencies, load data, train the model, and evaluate its performance.

### Best Model Configuration
[cite_start]The configuration below achieved the highest test accuracy on CIFAR-10. [cite: 16]

| Hyperparameter      | Value                |
| ------------------- | -------------------- |
| Patch Size          | e.g., 4x4 or 8x8     |
| Embedding Dimension | e.g., 512            |
| Number of Heads (MHSA)| e.g., 8              |
| Number of Layers    | e.g., 6              |
| MLP Hidden Dimension| e.g., 2048           |
| Optimizer           | e.g., AdamW          |
| Learning Rate       | e.g., 1e-4           |
| Weight Decay        | e.g., 0.05           |
| Batch Size          | e.g., 128            |
| Training Epochs     | e.g., 100            |
| Augmentations       | e.g., AutoAugment, Mixup |

### Results
[cite_start]The final classification accuracy on the CIFAR-10 test set is reported below. [cite: 16]

| Metric                          | Score     |
| ------------------------------- | --------- |
| **Overall Test Accuracy (%)** | **XX.XX%**|

### (Bonus) Analysis
[cite_start]Here is a brief analysis of design choices and their impact on performance. [cite: 18, 19]

* **Patch Size**: *(Example: I experimented with 4x4 and 8x8 patches. The smaller 4x4 patches resulted in a longer sequence of tokens but captured finer details, leading to a 2% accuracy improvement despite longer training times.)*
* **Augmentation Effects**: *(Example: Implementing RandAugment provided a significant boost over simple flips and crops, suggesting that strong regularization is crucial for ViT performance on smaller datasets like CIFAR-10.)*
* **Optimizer Choice**: *(Example: AdamW outperformed SGD with momentum, likely due to its better handling of weight decay.)*

---

## Question 2: Text-Driven Image Segmentation with SAM 2

[cite_start]This section covers the implementation of text-prompted image segmentation using SAM 2, as detailed in `q2.ipynb`. [cite: 21]

### How to Run (Q2)
1.  [cite_start]Open `q2.ipynb` in Google Colab. [cite: 24]
2.  Ensure the runtime is set to use a GPU accelerator.
3.  Run all cells from top to bottom. [cite_start]The notebook will install dependencies, load models, accept a text prompt, and display the final segmented image. [cite: 23, 24]

### Pipeline Description
[cite_start]My approach for text-prompted segmentation follows these steps: [cite: 25]
1.  **Load Image**: The user provides an image to be segmented.
2.  **Accept Text Prompt**: The user inputs a text string describing the object of interest (e.g., "a red car").
3.  [cite_start]**Generate Region Seeds**: I use **[Your Chosen Model, e.g., Grounding DINO]** to interpret the text prompt and generate bounding box coordinates (region seeds) corresponding to the object. [cite: 23]
4.  [cite_start]**Segment with SAM 2**: These bounding box seeds are fed into a pre-trained SAM 2 model. [cite: 23]
5.  [cite_start]**Display Mask**: SAM 2 outputs a segmentation mask for the object, which is then overlaid on the original image for visualization. [cite: 23]

### Limitations
[cite_start]The primary limitations of this pipeline are: [cite: 25]
* **Seed Generation Accuracy**: The final mask quality is highly dependent on the accuracy of the text-to-seed model (e.g., Grounding DINO). If it fails to locate the object correctly, SAM 2 will segment the wrong area.
* **Ambiguous Prompts**: Vague or ambiguous text prompts (e.g., "the person") can lead to incorrect or incomplete segmentation if multiple instances of the object exist.
* **Computational Cost**: Loading multiple large models (e.g., Grounding DINO and SAM 2) can be memory-intensive.

### (Bonus) Video Extension
*(If you completed the bonus, describe it here. Otherwise, you can delete this section.)*

I extended the pipeline to perform text-driven video object segmentation. The first frame is segmented using the text-prompt pipeline described above. [cite_start]For subsequent frames, the mask from frame `t-1` is used as a prompt to SAM 2 to segment frame `t`, allowing for mask propagation throughout the video clip. [cite: 27]