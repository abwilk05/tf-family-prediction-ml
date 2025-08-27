# Transcription Factor Family Prediction from Binding Specificity

Machine learning methods to predict transcription factor (TF) structural families directly from DNA-binding specificity motifs.

## Overview

This project demonstrates that TF structural families can be accurately predicted from binding specificity patterns encoded in position weight matrices (PWMs). We compare k-nearest neighbors (KNN) and deep learning approaches, achieving ~82-84% accuracy on 2,113 motifs across 20 TF families from JASPAR.

## Key Results

- **KNN Baseline**: 84.2% accuracy, 0.765 macro-F1
- **Deep Learning**: 82.2% accuracy, 0.785 macro-F1
- **Speed**: Neural network is ~1000x faster than KNN for predictions
- **Interpretability**: Deep model learns biologically meaningful features (e.g., CANNTG for bHLH, NTAATNN for homeodomain)

## Dataset

- Source: JASPAR 2024 database
- Size: 2,113 motifs across 20 TF families
- Format: Position weight matrices (4xN)
- Excluded: Dimers and rare families (<20 motifs)

## Methods

### K-Nearest Neighbors
- Distance metrics: Pearson correlation, Euclidean distance, Sandelin-Wasserman similarity
- Motif comparison via Tomtom tool
- Ensemble approach using majority voting

### Deep Learning
- Fully convolutional neural network
- Reverse complement invariant architecture
- Feature importance via Integrated Gradients
- Stratified k-fold cross-validation (k=5,10,20)

## Key Findings

1. **Performance Ceiling**: Both methods appear to approach an inherent limit set by binding specificity overlap between TF families
2. **Family-Specific Patterns**: Models successfully capture canonical recognition sequences
3. **Motif Similarity Analysis**: Family overlap metrics explain 61% (DL) and 45% (KNN) of per-class performance variance

## Repository Structure

```
tf-family-prediction-ml/
thesis-main/
├── CNN/                  # Deep learning implementation
│   ├── models/            
│   |   ├── FCN.py        # Fully convolutional network  
│   |   └── saves/        # Trained model checkpoints
│   ├── outputs/          # CNN results and predictions
│   ├── runs/             # Training runs and logs
│   ├── scripts/          # Core CNN scripts
│   │   ├── data_loader.py
│   │   ├── get_pwms.py
│   │   ├── helper.py
│   │   ├── plots_new.py
│   │   ├── plots.py
│   │   └── train_eval.py
│   └── main.py           # CNN main execution
├── KNN/                  # K-nearest neighbors implementation
│   ├── outputs/          # KNN results
│   ├── scripts/          # KNN scripts
│   │   ├── functions.py
│   │   ├── plots.py
│   └── main.py
├── other_analyses/        # Additional analyses
│   ├── consensus/        # Consensus sequence analysis
│   └── similarity_analysis/  # Motif similarity metrics
├── performance.txt       # Performance comparison results
└── Technical_Report.pdf  # Full thesis document
```
