# Datasets for Collaborative RLVR for Robust Reasoning

This directory contains datasets used for training and evaluating reinforcement learning with verifiable rewards (RLVR) systems for mathematical reasoning.

## Dataset Overview

| Dataset | Source (HuggingFace) | Splits | Size | Purpose |
|---------|---------------------|--------|------|---------|
| GSM8K | `openai/gsm8k` (config: `main`) | train (7,473), test (1,319) | 4.5 MB | Grade school math training & evaluation |
| MATH | `EleutherAI/hendrycks_math` (7 subjects) | train (7,500), test (5,000) | 9.6 MB | Competition math training & evaluation |
| GSM-Symbolic | `apple/GSM-Symbolic` | test (5,000) | 5.5 MB | Robustness evaluation (symbolic variants of GSM8K) |
| MATH-500 | `HuggingFaceH4/MATH-500` | test (500) | 416 KB | Standard evaluation benchmark (500-problem MATH subset) |
| DAPO-Math-17k | `open-r1/DAPO-Math-17k-Processed` | train (17,398) | 16 MB | RLVR training data from DAPO system |

**Total disk usage: ~36 MB**

---

## 1. GSM8K (Grade School Math 8K)

- **HuggingFace ID:** `openai/gsm8k` (config: `main`)
- **Paper:** Cobbe et al. (2021), "Training Verifiers to Solve Math Word Problems"
- **License:** MIT
- **Description:** 8,792 grade school math word problems. Each problem requires 2-8 steps of basic arithmetic. Answers include chain-of-thought solutions with a final numeric answer delimited by `#### <answer>`.
- **Columns:** `question`, `answer`
- **Splits:**
  - `train`: 7,473 examples
  - `test`: 1,319 examples

### Loading Instructions

```python
from datasets import load_dataset, load_from_disk

# From HuggingFace Hub
ds = load_dataset("openai/gsm8k", "main")

# From local disk
ds = load_from_disk("datasets/gsm8k/data")
```

---

## 2. MATH (Competition Mathematics)

- **HuggingFace ID:** `EleutherAI/hendrycks_math` (7 subject configs)
- **Paper:** Hendrycks et al. (2021), "Measuring Mathematical Problem Solving with the MATH Dataset"
- **License:** MIT
- **Description:** 12,500 competition mathematics problems spanning 7 subjects, each with difficulty levels 1-5. Includes full LaTeX solutions.
- **Columns:** `problem`, `level`, `type`, `solution`, `subject`
- **Subjects:** algebra (1,744/1,187), counting_and_probability (771/474), geometry (870/479), intermediate_algebra (1,295/903), number_theory (869/540), prealgebra (1,205/871), precalculus (746/546)
- **Splits:**
  - `train`: 7,500 examples (combined across subjects)
  - `test`: 5,000 examples (combined across subjects)

### Loading Instructions

```python
from datasets import load_dataset, load_from_disk

# From HuggingFace Hub (individual subject)
ds = load_dataset("EleutherAI/hendrycks_math", "algebra")

# From local disk (all subjects combined)
ds = load_from_disk("datasets/math/data")
```

---

## 3. GSM-Symbolic

- **HuggingFace ID:** `apple/GSM-Symbolic`
- **Paper:** Mirzadeh et al. (2024), "GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models"
- **License:** See dataset card
- **Description:** Symbolic variants of GSM8K problems where numerical values are systematically varied. Used to test robustness of mathematical reasoning -- whether models truly understand problem structure or rely on pattern matching.
- **Columns:** `id`, `instance`, `question`, `answer`, `original_id`, `original_question`, `original_answer`, `canary`
- **Splits:**
  - `test`: 5,000 examples (multiple symbolic variants per original GSM8K problem)
- **Note:** Contains a canary string indicating it should not appear in training data.

### Loading Instructions

```python
from datasets import load_dataset, load_from_disk

# From HuggingFace Hub
ds = load_dataset("apple/GSM-Symbolic")

# From local disk
ds = load_from_disk("datasets/gsm_symbolic/data")
```

---

## 4. MATH-500

- **HuggingFace ID:** `HuggingFaceH4/MATH-500`
- **Paper:** Lightman et al. (2023), "Let's Verify Step by Step" (used as evaluation set)
- **License:** See dataset card
- **Description:** A curated subset of 500 problems from the MATH benchmark, widely used as a standard evaluation benchmark in RLVR research. Problems span all difficulty levels and subjects.
- **Columns:** `problem`, `solution`, `answer`, `subject`, `level`, `unique_id`
- **Splits:**
  - `test`: 500 examples

### Loading Instructions

```python
from datasets import load_dataset, load_from_disk

# From HuggingFace Hub
ds = load_dataset("HuggingFaceH4/MATH-500")

# From local disk
ds = load_from_disk("datasets/math_500/data")
```

---

## 5. DAPO-Math-17k

- **HuggingFace ID:** `open-r1/DAPO-Math-17k-Processed` (processed version)
- **Original:** `BytedTsinghua-SIA/DAPO-Math-17k`
- **Paper:** Yu et al. (2025), "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (arXiv:2503.14476)
- **License:** See dataset card
- **Description:** 17,398 deduplicated math problems used for training in the DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization) system. This is a key training dataset for RLVR research, with problems sourced from various math competitions and datasets. Each entry includes the prompt, ground truth solution, and reward model configuration.
- **Columns:** `prompt`, `solution`, `data_source`, `source_prompt`, `ability`, `reward_model`, `extra_info`
- **Splits:**
  - `train`: 17,398 examples

### Loading Instructions

```python
from datasets import load_dataset, load_from_disk

# From HuggingFace Hub (processed/deduplicated version)
ds = load_dataset("open-r1/DAPO-Math-17k-Processed")

# From local disk
ds = load_from_disk("datasets/dapo_math_17k/data")
```

---

## Directory Structure

```
datasets/
├── README.md                          # This file
├── .gitignore                         # Excludes large data files from git
├── gsm8k/
│   └── data/                          # Arrow format (HuggingFace datasets)
├── math/
│   └── data/                          # Arrow format (all 7 subjects combined)
├── gsm_symbolic/
│   └── data/                          # Arrow format
├── math_500/
│   └── data/                          # Arrow format
├── dapo_math_17k/
│   └── data/                          # Arrow format
└── samples/                           # Small JSON samples (first 5 examples each)
    ├── gsm8k_sample.json
    ├── math_sample.json
    ├── gsm_symbolic_sample.json
    ├── math_500_sample.json
    └── dapo_math_17k_sample.json
```

## Re-downloading Datasets

To re-download all datasets from scratch:

```bash
source .venv/bin/activate
uv pip install datasets
python3 << 'EOF'
from datasets import load_dataset

# GSM8K
ds = load_dataset("openai/gsm8k", "main")
ds.save_to_disk("datasets/gsm8k/data")

# MATH (all subjects combined)
import datasets
subjects = ['algebra', 'counting_and_probability', 'geometry',
            'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
combined_train, combined_test = [], []
for subj in subjects:
    ds = load_dataset("EleutherAI/hendrycks_math", subj)
    for ex in ds["train"]: ex["subject"] = subj; combined_train.append(ex)
    for ex in ds["test"]: ex["subject"] = subj; combined_test.append(ex)
combined = datasets.DatasetDict({
    "train": datasets.Dataset.from_list(combined_train),
    "test": datasets.Dataset.from_list(combined_test),
})
combined.save_to_disk("datasets/math/data")

# GSM-Symbolic
ds = load_dataset("apple/GSM-Symbolic")
ds.save_to_disk("datasets/gsm_symbolic/data")

# MATH-500
ds = load_dataset("HuggingFaceH4/MATH-500")
ds.save_to_disk("datasets/math_500/data")

# DAPO-Math-17k
ds = load_dataset("open-r1/DAPO-Math-17k-Processed")
ds.save_to_disk("datasets/dapo_math_17k/data")
EOF
```

## Usage in Research

These datasets serve different roles in RLVR for robust reasoning research:

| Role | Datasets |
|------|----------|
| **RLVR Training** | DAPO-Math-17k, GSM8K (train), MATH (train) |
| **In-Distribution Evaluation** | GSM8K (test), MATH (test), MATH-500 |
| **Robustness Evaluation** | GSM-Symbolic |
| **Cross-Domain Transfer** | Train on one, evaluate on others |

The key research question is whether collaborative RLVR training (using multiple models or reward signals) produces more robust mathematical reasoning, as measured by performance on both in-distribution benchmarks and out-of-distribution robustness tests like GSM-Symbolic.
