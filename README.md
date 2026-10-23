# Knee MRI Abnormality Classification Using Deep Learning

Multi-label classification of knee MRI scans using DICOM image preprocessing, MobileNetV2 transfer learning, and TensorFlow/Keras.

## Project Overview

This project explores whether a pretrained convolutional neural network (CNN) can identify 12 knee abnormalities from MRI scans, including ACL and meniscus injuries, osteoarthritis, and joint effusion.

The project uses the RSNA Knee Abnormality Detection Kaggle competition dataset. Because expert-labeled data was limited, training incorporated LLM-generated pseudo-labels alongside expert annotations.

## Technologies

**Python:** pandas, NumPy, pydicom, Pillow  
**Deep Learning:** TensorFlow, Keras, MobileNetV2  
**Evaluation:** ROC-AUC, PR-AUC  
**Environment:** Kaggle Notebooks, GPU acceleration

## Methodology

### 1. MRI Preprocessing

- Selected one sagittal fluid-sensitive MRI series per study.
- Extracted three slices at approximately 20%, 50%, and 80% of the series.
- Applied intensity normalization and resized images to 160 × 160 pixels.
- Stacked the grayscale slices into a three-channel input.

### 2. Model Architecture

Used an ImageNet-pretrained MobileNetV2 backbone with global average pooling, a 256-unit dense layer, dropout, and 12 sigmoid output units.

Training was performed in two stages: first with the backbone frozen, followed by fine-tuning the final 40 backbone layers while keeping Batch Normalization layers frozen.

### 3. Training Strategy

The model was trained using 1,200 pseudo-labeled studies and 32 expert-labeled training studies.

Pseudo-labels were derived from LLM-generated scores. Uncertain labels were masked during binary cross-entropy loss calculation.

## Results

| Evaluation | Result |
|---|---|
| Expert-labeled validation (11 studies) | 0.690 macro ROC-AUC |
| Weak-label evaluation (300 studies) | ~0.663 macro ROC-AUC |
| Kaggle public leaderboard | 0.631 public score |

Fine-tuning improved results relative to the frozen-backbone baseline. However, the small expert-labeled validation set limits confidence in the estimated performance.

The weak-label evaluation measures agreement with LLM-generated labels, not independently verified clinical accuracy. The Kaggle public score is reported separately because its evaluation setting may differ from local validation.

## Limitations

- Only 58 studies had expert annotations across the original expert-labeled splits.
- Expert validation involved just 11 studies.
- Pseudo-labels may introduce labeling errors.
- Three selected slices do not represent the full MRI volume.
- The model has not been clinically validated and is not intended for diagnostic use.

## Notebooks

**[Training and Evaluation](notebooks/01_mri_model_training_evaluation.ipynb)** — Data preprocessing, model training, fine-tuning, and evaluation.

**[Kaggle Submission](notebooks/02_kaggle_submission.ipynb)** — Loading trained weights, generating test predictions, and creating the submission file.

## Reproducibility

Training and inference were performed in Kaggle using competition data and GPU acceleration. The training notebook retains saved outputs and was not rerun after portfolio documentation cleanup.

The inference notebook loads previously trained weights without retraining. Reproduction requires access to the competition data, pseudo-label dataset, and saved model weights.

## Key Takeaways

This project provided practical experience with medical imaging, DICOM preprocessing, transfer learning, multi-label classification, model evaluation, and working with limited ground-truth data.
