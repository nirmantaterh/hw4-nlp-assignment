# CS2590 HW4: BERT Sentiment Classification & T5 Text-to-SQL

## Overview
This repository contains code and results for NYU CS2590 NLP Assignment 4, Spring 2026.

**Student:** Nirman Taterh (nt2613)

---

## Part I: BERT Fine-tuning for Sentiment Classification (IMDB)

### Q1: Fine-tuning BERT Model
- **Task:** Fine-tune BERT on IMDB dataset for binary sentiment classification
- **Result:** 91.74% test accuracy (threshold: >91%) ✓
- **Training:** 3 epochs on full IMDB training set
- **Files:** `main.py`, `utils.py`

### Q2: Data Transformation (QWERTY Typos)
- **Task:** Design & implement realistic out-of-distribution transformation
- **Approach:** Character-level QWERTY keyboard proximity typo injection (10% corruption rate)
- **Rationale:** Simulates common user typing errors while preserving sentiment labels
- **Result:** 75.56% accuracy on transformed test set (drop: 16.18 points) ✓
- **Grade:** Full credit (drop > 4 points)

### Q3: Data Augmentation Analysis
- **Task:** Augment training data with transformed examples; evaluate robustness
- **Training:** 3 epochs with 5,000 augmented examples added to training set
- **Results:**
  - Augmented model on original test: 90.63% (vs 91.74% non-augmented)
  - Augmented model on transformed test: 84.71% (vs 75.56% non-augmented)
  - Improvement on OOD data: +9.15 points ✓
  - Trade-off on ID data: -1.11 points (acceptable)
- **Insight:** Augmentation trades in-distribution accuracy for out-of-distribution robustness
- **Files:** `train_and_save.py`, `main.py`

---

## Part II: T5 Fine-tuning for Text-to-SQL (Spider Dataset)

### Q4-Q5: Data & Architecture
- **Model:** T5-small (60M parameters)
- **Task:** Generate SQL queries from natural language descriptions
- **Data:** 4,225 training, 466 dev examples
- **No preprocessing applied** (used as-is)

### Q6-Q7: Results & Error Analysis
- **Test Performance:** Record F1 = 73.35% (threshold: >65%) ✓
- **Dev Performance:** Record F1 = 76.32%, Query EM = 1.93%
- **Best Checkpoint:** Epoch 41 (saved to Google Drive)

**Error Breakdown (Development Set):**
- SQL Syntax Errors: 30/466 (6.4%)
- Column/Table Not Found: 15/466 (3.2%)
- Query Timeouts: 7/466 (1.5%)
- Valid SQL queries: 414/466 (88.8%)
- Semantic correctness gap: 12.5% (valid SQL but wrong results)

**Files:** `train_t5.py`, `load_data.py`, `t5_utils.py`

---

## Key Hyperparameters

### BERT (Part I)
- Learning Rate: 2e-5
- Batch Size: 16
- Epochs: 3
- Optimizer: AdamW
- Loss: CrossEntropyLoss

### T5 (Part II)
- Learning Rate: 1e-4
- Batch Size: 16 (training), 32 (eval)
- Epochs: 45 (with early stopping, patience=10)
- Optimizer: AdamW (weight_decay=0.01)
- Learning Rate Warmup: 2 epochs linear, then cosine annealing
- Decoding Strategy: Progressive (greedy → sampling → beam search)

---

## Model Checkpoints

**T5 Best Model (Epoch 41):**
- Link: [Google Drive](https://drive.google.com/drive/folders/1gEUdtCcUQFuFTCGe4VxGRXcH8WJPP1TU?usp=sharing)
- Files: `model_best.pt`, `config.json`

---

## File Structure
