
# Arabic Sign Language Recognition using Deep Learning

This project presents a deep learning framework for **Arabic Sign Language (ARSL) recognition**, evaluating multiple models under different data balancing strategies and enhancing performance through probability-based ensemble learning and statistical analysis.

---

## 📌 Dataset

The study uses the **Arabic Sign Language Dataset (ARSL2018)** available on Mendeley Data:

🔗 https://data.mendeley.com/datasets/y7pckrw6z2/1
The processed dataset files are large and are hosted externally.


It contains:
- 54,049 labeled images  
- 32 sign language classes  
- ~40 participants  
- Variations in lighting, background, and hand pose  

---

## 📁 Project Structure

```plaintext
Arabic-Sign-Language/
│
├── data/
│   ├── processed/
│   │   ├── original/
│   │   │   ├── images_original.npy
│   │   │   └── encoded_labels_original.npy
│   │   │
│   │   ├── undersampling/
│   │   │   ├── images_under.npy
│   │   │   └── encoded_labels_under.npy
│   │   │
│   │   ├── oversampling/
│   │   │   ├── images_over.npy
│   │   │   └── encoded_labels_over.npy
│   │   │
│   │   └── label_encoder_classes.npy  ← (GLOBAL / SHARED)
│   │
│   ├── ARSL2018/
│   │   └── dataset_link.txt
│   │
│   └── (optional raw / future datasets)
│
├── notebooks/
│   ├── CNN/
│   ├── MobileNetV2/
│   └── CNN_LSTM/
│
├── probabilities_notebooks/
│   ├── cnn_probabilities.ipynb
│   └── mobilenetv2_probabilities.ipynb
│
├── ensemble/
│   ├── ensemble_original.ipynb
│   ├── ensemble_undersampling.ipynb
│   └── ensemble_oversampling.ipynb
│
├── statistical_analysis/
│   ├── per_model_analysis.ipynb
│   └── model_comparison_analysis.ipynb
│
├── requirements.txt
├── README.md
└── CITATION.cff
```

---

## ⚙️ Methodology Overview

The framework consists of three main stages:

### 1. Model Training
Three deep learning architectures are trained independently:
- CNN
- MobileNetV2 (Transfer Learning)
- CNN-LSTM hybrid model

Each model is evaluated under three dataset configurations:
- Original
- Undersampling
- Oversampling

---

### 2. Probability-Based Ensemble
To improve prediction performance, class probabilities are extracted from:
- CNN
- MobileNetV2

These probabilities are then combined in separate ensemble experiments for each dataset configuration.

---

### 3. Statistical Evaluation
A comprehensive analysis is performed at two levels:
- **Per-model evaluation** across datasets
- **Cross-model comparison** between CNN, MobileNetV2, and CNN-LSTM

---

## 📊 Evaluation Metrics

- Accuracy  
- Loss  
- Matthews Correlation Coefficient (MCC)  
- ROC-AUC
- F1-Weighted
- Precision-Weighted
- Recall-Weighted
- Training Time (s)
- Memory Used (MB)

---

## 📈 Pipeline Visualization
