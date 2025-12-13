# Final Project for CPSC 5710: Building Trustworthy Vision Transformers
## 3. Vision Transformer Robustness for Patch Based Adversarial Attacks & Defenses 
We implement a robustness evaluation pipeline for Vision Transformers (ViT, DeiT, DINOv2 models) using the "ILSVRC/imagenet-1k" dataset in `patch-attack-defenses-final.ipynb`. Each section of the notebook has detailed comments on how to reproduce and run the pipeline for evaluations and capture the experiment results  
The main sections are:
* Setup: Authenticate with Hugging Face, install dependencies, and load models with ImageNet-1k label mappings.
* Baseline Generation: Stream the validation split dataset to select MAX_SAMPLES images that are correctly classified by the target models (Clean Baseline).
* Adversarial Experimentation: For each baseline image, generate adversarial variants using patch attacks (Token, Patch-Perturbation, Patch-Fool) and apply input defenses (Blur, Compression, Patch Masking). Iterate through the baseline images to generate a detailed dataset of adversarial and defended variants, performing inference to capture predictions and probabilities.
* Metric Calculation: Compute aggregate metrics for Accuracy and Attack Success Rate (ASR) based on the model's predictions to evaluate defense effectiveness
* Publishing Results: Save the detailed results (images, labels, predictions) and an aggregate summary table as datasets to the Hugging Face Hub.
