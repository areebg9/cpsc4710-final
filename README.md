# Final Project for CPSC 5710: Building Trustworthy Vision Transformers

## 1. Adversarial Training
In the `adversarial-training` folder, the notebook runs through the full adversarial training pipeline and baseline evaluations. Each section is explained alongside the associated code. The main sections are (i) adversarial training, which is adapted from our previous written homework but adjusted for the DINOv2 pipeline, and (ii) overall evaluations which loads in the robustness token models from HuggingFace and compares these models across the central adversarial dataset.

In various parts of the code, large datasets on the orders of 10s or 100s of GBs would need to be loaded for the full training run. To ensure reproducibility, there are some changes in the notebook that enable the user to selectively download certain training files, pre-load our trained models, and limit the evaluations to a fewer number of samples.

Before running the code, remember to set `HF_TOKEN` as an environment variable to be used for dataset and model loading.

## 2. Robustness Tokens

We provide our fork of Pulfer et al.’s repository at [https://github.com/rohanphanse/robustness-tokens](https://github.com/rohanphanse/robustness-tokens). The training and evaluation results in Figure 7 and Table 2 can be reproduced by following the setup guide in [README.md](https://github.com/rohanphanse/robustness-tokens/blob/main/README.md) and then running [train.sh](https://github.com/rohanphanse/robustness-tokens/blob/main/train.sh) and [eval.sh](https://github.com/rohanphanse/robustness-tokens/blob/main/eval.sh) respectively.

The trained weights for robustness-token backbones (with 10 robustness tokens) and their corresponding custom linear heads for DINOv2-ViTS/14, DINOv2-ViTB/14, and DINOv2-ViTL/14 are available at https://huggingface.co/cpsc-5710-final-vit-robustness/robustness-token-trained-models/tree/main. These weights can be used to reproduce the robustness-token results reported in Tables 2 and 3.

## 3. Vision Transformer Robustness for Patch Based Adversarial Attacks & Defenses 
We implement a robustness evaluation pipeline for Vision Transformers (ViT, DeiT, DINOv2 models) using the "ILSVRC/imagenet-1k" dataset in `patch-attack-defenses-final.ipynb`. Each section of the notebook has detailed comments on how to reproduce and run the pipeline for evaluations and capture the experiment results  
The main sections are:
* Setup: Authenticate with Hugging Face, install dependencies, and load models with ImageNet-1k label mappings.
* Baseline Generation: Stream the validation split dataset to select MAX_SAMPLES images that are correctly classified by the target models (Clean Baseline).
* Adversarial Experimentation: For each baseline image, generate adversarial variants using patch attacks (Token, Patch-Perturbation, Patch-Fool) and apply input defenses (Blur, Compression, Patch Masking). Iterate through the baseline images to generate a detailed dataset of adversarial and defended variants, performing inference to capture predictions and probabilities.
* Metric Calculation: Compute aggregate metrics for Accuracy and Attack Success Rate (ASR) based on the model's predictions to evaluate defense effectiveness
* Publishing Results: Save the detailed results (images, labels, predictions) and an aggregate summary table as datasets to the Hugging Face Hub.
