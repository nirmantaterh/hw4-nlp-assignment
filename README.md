# CS2590 Assignment 4: Fine Tuning LMs

## Overview
This repository contains code and results for NYU CS2590 NLP Assignment 4, Spring 2026.

**Student:** Nirman Taterh (nt2613)

---

## Part I: BERT Fine-tuning for Sentiment Classification (IMDB)

### Q1: Fine-tuning BERT Model
- **Task:** Fine-tune BERT on IMDB dataset for binary sentiment classification
- **Result:** 91.74% test accuracy (threshold: >91%) 
- **Training:** 3 epochs on full IMDB training set
- **Files:** `main.py`, `utils.py`

### Q2: Data Transformation (QWERTY Typos)
- **Task:** Design & implement realistic out-of-distribution transformation
- **Approach:** Character-level QWERTY keyboard proximity typo injection (10% corruption rate)
- **Rationale:** Simulates common user typing errors while preserving sentiment labels
- **Result:** 75.56% accuracy on transformed test set (drop: 16.18 points) 
- **Grade:** Full credit (drop > 4 points)

### Q3: Data Augmentation Analysis
- **Task:** Augment training data with transformed examples; evaluate robustness
- **Training:** 3 epochs with 5,000 augmented examples added to training set
- **Results:**
  - Augmented model on original test: 90.63% (vs 91.74% non-augmented)
  - Augmented model on transformed test: 84.71% (vs 75.56% non-augmented)
  - Improvement on OOD data: +9.15 points 
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
- **Test Performance:** Record F1 = 73.35% (threshold: >65%) 
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
```hw4-nlp-assignment/
├── main.py                      # BERT training & evaluation
├── utils.py                     # QWERTY typo transformation, data loading
├── train_and_save.py            # Augmented model training
├── train_t5.py                  # T5 fine-tuning pipeline
├── load_data.py                 # Data loading utilities
├── t5_utils.py                  # T5-specific functions
├── hw4_final_updated.pdf        # Written report
├── out_original.txt             # Q1 predictions
├── out_transformed.txt          # Q2 predictions
├── out_augmented_original.txt   # Q3 original predictions
├── out_augmented_transformed.txt # Q3 transformed predictions
├── t5_ft_experiment_test.sql    # T5 test predictions (SQL)
├── t5_ft_experiment_test.pkl    # T5 test predictions (pickled)
└── README.md                    # This file

---

## AI Usage Documentation

**Tools Used:** Claude (Anthropic), NotebookLM

**Usage:**
1. Understanding assignment requirements and clarifying ambiguous sections
2. Debugging code errors and suggesting fixes
3. Analyzing model errors and categorizing by error type

**Note:** All code was authored by me. AI provided explanations and guidance; all implementation and experimental design decisions were made independently.
```
---

## Results Summary

| Component | Metric | Result | Threshold | Status |
|-----------|--------|--------|-----------|--------|
| Q1 BERT | Test Accuracy | 91.74% | >91% |  PASS |
| Q2 Transform | Accuracy Drop | 16.18 pts | >4 pts |  PASS |
| Q3 Augment | OOD Improvement | +9.15 pts | N/A |  WRITTEN |
| Q7 T5 | Record F1 | 73.35% | >65% |  PASS |

