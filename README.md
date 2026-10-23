# Knee MRI Abnormality Classification Using Transfer Learning

## Project Overview

This project explores multi-label classification of knee MRI scans using deep learning. The goal was to develop a computer vision pipeline that processes DICOM MRI images and predicts 12 knee abnormalities, including ligament injuries, meniscus injuries, osteoarthritis, and joint effusion.

The project uses the RSNA Knee Abnormality Detection Kaggle dataset and MobileNetV2 transfer learning.

## Tools & Technologies

- **Python:** NumPy, pandas, pydicom, Pillow
- **Deep Learning:** TensorFlow, Keras, MobileNetV2
- **Evaluation:** ROC-AUC, PR-AUC
- **Environment:** Kaggle Notebooks with GPU acceleration

## Methodology

**1. Data Preprocessing**

- Selected a sagittal fluid-sensitive MRI series for each study.
- Extracted three slices at approximately 20%, 50%, and 80% of the series.
- Normalized image intensities and resized slices to 160 × 160.
- Stacked the three grayscale slices into a three-channel image.

**2. Model Development**

- Used ImageNet-pretrained MobileNetV2 as the feature extraction backbone.
- Initially froze the backbone to train the classification head.
- Fine-tuned the final 40 backbone layers, excluding Batch Normalization layers.
- Used sigmoid outputs to predict 12 abnormalities independently.

**3. Training**

Because expert-labeled training data was limited, the model used a combination of expert labels and LLM-generated pseudo-labels. Uncertain pseudo-labels were masked during loss calculation.

**4. Evaluation**

| Evaluation | Result |
|---|---|
| Expert-labeled validation (11 studies) | 0.690 macro ROC-AUC |
| Weak-label evaluation (300 studies) | Approximately 0.663 macro ROC-AUC |
| Kaggle public leaderboard | 0.631 public score |

## Key Findings

Fine-tuning improved performance compared with the frozen-backbone baseline. However, the limited expert-labeled validation set introduces substantial uncertainty.

The evaluation against LLM-generated labels measures agreement with pseudo-labels rather than independently verified clinical accuracy.

## Limitations

- Only 58 expert-labeled studies were available across the original expert-labeled splits.
- Validation results are based on a small sample.
- Three selected slices do not represent the full MRI volume.
- Pseudo-labels may contain labeling errors.
- The model is a research and portfolio demonstration, not a clinically validated diagnostic system.

## Reproducibility

Training and inference were performed in Kaggle using competition data and GPU acceleration. The portfolio notebook retains saved execution outputs; it was not fully rerun after documentation cleanup.

The inference notebook reconstructs the model architecture, loads saved weights, and generates competition predictions without retraining.

## Project Takeaways

This project provided hands-on experience with medical image preprocessing, transfer learning, fine-tuning pretrained CNNs, multi-label classification, and evaluating models with limited ground-truth labels.
