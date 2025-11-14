# Breaking CAPTCHA with CRNN  
*A modern deep learning approach to one of the internet’s first AI benchmarks*  

**Author:** Emilio Sanchez-Harris  
**Course:** CS 5100 — Northeastern University  

---

## Overview  
This project implements a Convolutional Recurrent Neural Network (CRNN) for breaking text-based CAPTCHA systems.  
The model combines a CNN feature extractor, a bidirectional LSTM sequence model, and a Connectionist Temporal Classification (CTC) loss function to perform end-to-end sequence recognition directly from images.

The system achieves:

- 90% validation accuracy  
- 88% test-set CAPTCHA-level accuracy  
- 98.6% character-level accuracy  

The aim is both to demonstrate how vulnerable classical CAPTCHA systems are to modern AI models, and to analyze failure cases that may inform stronger CAPTCHA design.

---

## Features

- Custom CRNN implementation using PyTorch  
- End-to-end training with CTC loss  
- OpenCV preprocessing: Gaussian blur, thresholding, morphological cleanup  
- Automatic dataset splitting into train/validation/test  
- Full training pipeline with learning rate scheduling and early stopping  
- Evaluation metrics including accuracy, confidence, and character-level errors  
- Training curve visualization  
- Prediction decoding and per-sample display  

---

## Results  

| Metric | Score |
|--------|--------|
| CAPTCHA-Level Accuracy | 88.0% |
| Character-Level Accuracy | 98.6% |
| Average Error Confidence | 99.3% |

Most Common Misclassifications:  
I → l, I → 1, l → i  

Most Error-Prone Characters:  
1, 7, 4, 0, 5, 9, I, 8, 3, 2  

These results show that CRNN models excel at text recognition but consistently struggle with visually ambiguous characters. This suggests both potential improvements to the model and insights for designing more robust CAPTCHA systems.

---

## Architecture  

The model consists of:

1. CNN Feature Extractor  
   Learns spatial patterns such as edges, curves, and eventually full character shapes using stacked convolutional layers and pooling.

2. Sequence Transformation  
   The final feature map is reshaped into a sequence of 25 time steps, each containing 1536 features.

3. Bidirectional LSTM  
   Captures contextual relationships between neighboring characters in both forward and backward directions.

4. CTC Loss  
   Allows alignment-free training by marginalizing over all valid sequences and inserting a blank token for handling irregular spacing.

The architecture is trained end-to-end using Adam and ReduceLROnPlateau learning rate scheduling.

---

## Setup and Usage

### 1. Install dependencies

CPU version:
```
pip install numpy opencv-python matplotlib scikit-learn tqdm torch torchvision
```

GPU (CUDA 11.8):
```
pip install numpy opencv-python matplotlib scikit-learn tqdm torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

GPU (CUDA 12.1):
```
pip install numpy opencv-python matplotlib scikit-learn tqdm torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Run the training script  
Open the project folder in your editor of choice and run:

```
python AIFinal.py
```

### 3. Example output  
```
Found 10000 images
Training on: cuda
...
Training complete! Best validation accuracy: 0.9000

=== Test Set Results ===
CAPTCHA-level Accuracy: 0.8800
Character-level Accuracy: 0.9866
Average Confidence: 0.9977
```

Training curves and prediction samples are saved in the `output` directory.

---

## Visualization  

A training curve figure is included in the repository as `training_curves.png`.

![](https://github.com/EmilioSzH/AIFinal/blob/main/output/training_curves.png)

---

## Dataset  

Source: Aadhav Vignesh — CAPTCHA Images (Kaggle)  
A dataset of 10,000 CAPTCHA images, each labeled with the correct text as the filename.

---

## Insights  

- CRNN architectures are highly effective OCR systems even for distorted text.  
- High-confidence errors involving characters such as I, l, and 1 reveal consistent blind spots.  
- Numerical digits contributed disproportionately to misclassifications.  
- Potential improvements include:
  - Separate digit/alphabet classifiers  
  - Character-specific data augmentation  
  - Hard-negative mining  

For CAPTCHA design, increasing ambiguity (e.g., mixing visually similar characters) or adding more varied symbol types may reduce the success rate of AI-based attacks.

---

## References  

1. Broder, Andrei Z., et al. “Method for Selectively Restricting Access to Computer Systems.” 2001.  
2. Shi, Baoguang, Xiang Bai, and Cong Yao. “An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition.” IEEE TPAMI, 2016.  
3. Graves, Alex et al. “Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks.” ICML, 2006.  
4. Vignesh, Aadhav. “Captcha Images.” Kaggle, 2020.  
