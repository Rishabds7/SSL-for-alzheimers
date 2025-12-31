# SSL-for-alzheimers
Using SSL model to understand Alzheimers/Tumors Pattern through MRI can perform classification and Providing visual assistance to clinicians to study the  MRI


# Self-Supervised MRI Representation Learning for Alzheimers and Cross-Domain Brain Diseases Classification

## Overview and Purpose
This repository implements a complete workflow for **Self-Supervised Learning (SSL)** applied to brain MRI classification using **SimCLR**.  
The goal is to pretrain a neural network on a large set of **unlabeled MRI scans** and fine-tune it on smaller labeled datasets for multiple neurological diseases, including:

- Alzheimer’s Disease  
- Brain Tumor  
- Parkinson’s Disease  
- Multiple Sclerosis  

The motivation is to leverage SSL to extract meaningful features from medical data, which is often difficult and costly to label. This enables improved generalization and performance across different brain disease domains with limited supervision.

---

## Step-by-Step Functionality

### 1. Data Loading and Preparation
- MRI images are collected and organized by disease type and class.  
- Approximately **1,800 labeled** and **100,000 unlabeled** samples are used.  
- Images are resized to **128×128 pixels** and normalized.  
- Data augmentations applied include random rotation, flips, and color jitter to improve robustness.  
- The dataset is split into **training (70%)**, **validation (15%)**, and **testing (15%)** sets.

### 2. Self-Supervised Pretraining with SimCLR
- The **SimCLR** framework learns image representations through **contrastive learning**.  
- Each MRI image undergoes two random augmentations, forming **positive pairs**, while other images act as **negative pairs**.  
- The objective is to maximize agreement between positive pairs and minimize it for negatives using the **NT-Xent contrastive loss**.

**Process Summary:**

1. Load unlabeled MRI images.  
2. Generate two augmented versions per image.  
3. Pass both through an **encoder** (ResNet-50 backbone).  
4. Map features into a latent space via a **projection head**.  
5. Compute contrastive loss and update encoder weights.

This pretraining phase helps the model learn structural and texture-level MRI features transferable across diseases.

### 3. Fine-Tuning on Downstream Tasks
After pretraining, the encoder is fine-tuned on small labeled datasets for each classification task:
- **Brain Tumor Classification**
- **Parkinson’s Disease Detection**
- **Multiple Sclerosis Diagnosis**
- **Alzheimer’s Disease Staging**

Each task uses the pretrained encoder with an added **fully connected classifier head** trained using cross-entropy loss.  
Fine-tuning improves task-specific performance while preserving general features learned during SSL.

### 4. Evaluation Metrics
Performance of each fine-tuned model is evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix (for per-class performance)

Results show that SimCLR significantly outperforms a supervised ResNet baseline, particularly when labeled data is scarce.

### 5. Model Explainability with Grad-CAM
To interpret model predictions, **Grad-CAM** (Gradient-weighted Class Activation Mapping) is applied.  
It highlights regions of MRI scans that most influence model decisions, confirming whether predictions are medically relevant.

Additional interpretability metrics include:
- Activation Intensity  
- Activation Area Percentage  
- Class Confidence

---

## Technical Implementation

### Key Libraries Used

| Library | Purpose |
|----------|----------|
| **PyTorch / Torchvision** | Core framework for model definition and training |
| **NumPy / Pandas** | Numerical computation and data handling |
| **Matplotlib / Seaborn** | Visualization of losses, metrics, and results |
| **scikit-learn** | Evaluation metrics and dataset utilities |
| **OpenCV / PIL** | Image preprocessing and augmentation |
| **Grad-CAM** | Explainable AI visualization tool |

### Model Architecture

The implemented SimCLR model consists of the following components:

1. **Base Encoder (ResNet-50)** – Extracts visual features from MRI images.  
2. **Projection Head (MLP)** – Projects encoded features into a latent space for contrastive loss computation.  
3. **Contrastive Learning Module** – Optimizes representations using NT-Xent loss.  
4. **Classifier Head (Fine-tuning Stage)** – A simple dense layer for downstream classification.

**Architecture Workflow:**

```
Input MRI → Encoder (ResNet-50) → Projection Head → Contrastive Loss
                     ↓
              Fine-tuned Classifier
```

---

## Results Summary
After fine-tuning on four downstream tasks:
- SimCLR achieved **over 90% accuracy** across all datasets.  
- The approach performed best on Alzheimers, Brain Tumor and Parkinson’s datasets.  
- SSL-based features generalized effectively to unseen diseases while requiring fewer labeled samples.

---

## Reference
For the full project write up, visualizations, and results, visit the official project website:  
https://huggingface.co/spaces/Piyushg7/AI4Alzheimers

**Datasets Used:**

**1) Alzheimer**

https://www.kaggle.com/datasets/ninadaithal/imagesoasis

**2) Brain Tumor**

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

**3) Parkinson**

https://www.kaggle.com/datasets/irfansheriff/parkinsons-brain-mri-dataset


**4) Multiple Sclerosis**

https://www.kaggle.com/datasets/orvile/multiple-sclerosis-brain-mri-lesion-segmentation


