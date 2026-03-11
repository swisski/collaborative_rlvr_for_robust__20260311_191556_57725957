# GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models

**Paper:** Mirzadeh, Alizadeh, Shahrokhi, Tuzel, Bengio, Farajtabar (Apple)
**ArXiv:** 2410.05229v1, October 7, 2024
**Pages:** 22 (12 main + 10 appendix)

---

## 1. Overview and Motivation

The paper investigates whether the high GSM8K scores reported by modern LLMs reflect genuine mathematical reasoning or sophisticated pattern matching. The authors introduce **GSM-Symbolic**, an improved benchmark built from symbolic templates derived from GSM8K, enabling controllable and reproducible evaluations. Their large-scale study across 25 models reveals that LLM mathematical "reasoning" is fragile, highly sensitive to superficial changes, and likely based on probabilistic pattern matching rather than formal logical reasoning.

### Core Problems with GSM8K
- **Static and fixed:** Single test set with single-point accuracy metric; no ability to control difficulty or generate variants.
- **Data contamination risk:** Popularity means test examples may have leaked into training data. The paper finds that for 21 out of 25 models, GSM8K performance falls on the right (optimistic) side of the GSM-Symbolic performance distribution, consistent with contamination.
- **No controllability:** Cannot test how models respond to changes in difficulty, variable values, names, or irrelevant information.

---

## 2. How GSM-Symbolic Is Constructed

### Template Creation Process (Section 3.1)
Starting from specific GSM8K test set examples, the authors create **parsable symbolic templates**:

1. **Identify variables:** Numerical values and proper names in each question are replaced with symbolic placeholders (e.g., `{x}`, `{name}`, `{family}`).
2. **Define variable domains:** Each variable gets a sampling range or set (e.g., `x = range(5, 100)`, `name = sample(names)`, `family = sample(["nephew", "cousin", "brother"])`).
3. **Specify conditions:** Constraints ensure correctness (e.g., divisibility for whole-number answers, `x + y + z + ans == total`).
4. **Template both question and solution:** The answer derivation is also parameterized so that generated instances come with correct solutions.

### Example (Figure 1)
- **Original GSM8K:** "Sophie watches her nephew... bag has 31 blocks... bin has 8 stuffed animals... tower has 9 rings... total 62... how many bouncy balls?"
- **Template:** `{name}` watches her `{family}`... bag has `{x}` blocks... bin has `{y}` stuffed animals... tower has `{z}` rings... total `{total}`... with condition `x + y + z + ans == total`.

### Quality Assurance
- **Automated checks:** Verify no original values appear in templates; verify original values satisfy all conditions; verify final answer matches.
- **Manual review:** 10 random samples per template reviewed manually.
- **Cross-model validation:** After evaluating all models, verify that at least 2 models answer each question correctly; otherwise, manual review.

### Scale
- **100 templates** selected from GSM8K test set.
- **50 samples per template** = 5,000 total examples per benchmark variant.
- This yields **50 datasets of 100 examples each**, where each dataset is one instantiation of all 100 templates.

---

## 3. Variations Tested

### 3a. GSM-Symbolic (Base)
Both names and numerical values changed simultaneously across 50 instantiations per template.

### 3b. Name-Only Changes
Only proper names (persons, places, foods, currencies) are changed; numerical values remain the same as the original GSM8K question.

### 3c. Number-Only Changes
Only numerical values are changed; proper names remain the same.

### 3d. Difficulty Variations (Section 4.3, Figure 5)
Templates are modified to adjust the number of reasoning clauses:
- **GSM-Symbolic-M1 (GSM-M1):** Remove one clause (easier).
- **GSM-Symbolic (base):** Original number of clauses.
- **GSM-Symbolic-P1 (GSM-P1):** Add one clause (harder).
- **GSM-Symbolic-P2 (GSM-P2):** Add two clauses (hardest).

Example: A phone call pricing problem starts with 2 price tiers; P1 adds a third tier; P2 adds the third tier plus a discount condition.

### 3e. GSM-NoOp (Section 4.4)
**Most novel and impactful variant.** Seemingly relevant but ultimately irrelevant ("No-Op") statements are added to GSM-Symbolic templates. These additions carry no operational significance and should not affect the reasoning chain.

Example: "Oliver picks 44 kiwis on Friday. Then he picks 58 kiwis on Saturday. On Sunday, he picks double the number of kiwis he did on Friday, **but five of them were a bit smaller than average**. How many kiwis does Oliver have?" -- The size of the kiwis is irrelevant to the count, but models subtract 5.

---

## 4. Models Tested

### Open Models (20+, sizes 2B to 27B)
- **Gemma family:** Gemma-2b, Gemma-2b-it, Gemma-7b, Gemma-7b-it, Gemma2-2b, Gemma2-2b-it, Gemma2-9b, Gemma2-9b-it, Gemma2-27b-it
- **Phi family:** Phi-2, Phi-3-mini-128k-instruct, Phi-3-small-128k-instruct, Phi-3-medium-128k-instruct, Phi-3.5-mini-instruct
- **Mistral family:** Mistral-7b-v0.1, Mistral-7b-instruct-v0.1, Mistral-7b-v0.3, Mistral-7b-instruct-v0.3, Mathstral-7b-v0.1
- **Llama family:** Llama3-8b, Llama3-8b-instruct

### Closed Models
- GPT-4o-mini
- GPT-4o
- o1-mini
- o1-preview

### Evaluation Setup
- **~500 total evaluations** across all setups.
- **8-shot Chain-of-Thought (CoT) prompting** with greedy decoding (except o1 models which do not support greedy decoding via API).
- Prompt format: system instruction ("As an expert problem solver...") followed by 8 Q/A pairs with "Let's think step by step" format, then the target question.
- Shots come from original GSM8K unless otherwise specified (NoOp experiments test alternative shot sources).

---

## 5. Key Findings

### 5.1 Performance Drops on GSM-Symbolic (Section 4.1)

**All models show lower average performance on GSM-Symbolic compared to GSM8K.**

Selected drops (GSM8K -> GSM-Symbolic):
| Model | GSM8K | GSM-Symbolic | Drop |
|-------|-------|-------------|------|
| Mistral-7b-it-v0.1 | 42.0 | 30.5 | -9.2 |
| Gemma2-2b | 46.0 | 40.1 | -7.4 |
| Gemma2-9b-it | 87.0 | 79.1 | -6.2 |
| Phi-3-medium | 89.0 | 82.5 | -4.8 |
| Phi-3.5-mini | 88.0 | 82.1 | -2.8 |
| GPT-4o-mini | 95.0 | 91.7 | -2.4 |
| GPT-4o | 95.0 | 94.9 | -0.3 |
| o1-mini | 93.0 | 94.5 | -0.6 |
| o1-preview | 96.0 | 92.7 | -2.2 |

**High variance across instantiations:**
- Gemma2-9B: gap between worst and best performance > 12%.
- Phi-3.5-mini: gap ~15%.
- Even GPT-4o shows variance of +/-1.9%.

**Evidence of data contamination:** For 21/25 models, GSM8K performance is >1 standard deviation to the right of the GSM-Symbolic distribution center. Models with the highest drops (Gemma2-9B, Phi-3, Mathstral) show the most right-shifted GSM8K performance.

### 5.2 Sensitivity to Different Change Types (Section 4.2)

**Name changes vs. number changes vs. both:**

Example (Gemma2-9b-it):
| Change Type | Mean Accuracy | Std Dev |
|-------------|--------------|---------|
| GSM8K (original) | 87.0 | - |
| Names only | 88.6 | +/-2.0 |
| Numbers only | 83.1 | +/-2.2 |
| Both | 79.1 | +/-3.0 |

**Key observations:**
- Models are **more robust to name changes** than number changes.
- GSM8K accuracy is much closer to the name-only distribution center (further evidence of contamination for specific numerical values).
- **Variance increases** progressively: names < numbers < both.
- Even name-only changes produce noticeable variance, which "would not be expected from a grade-school student with genuine mathematical understanding."

### 5.3 Difficulty Scaling (Section 4.3)

As the number of clauses increases from M1 -> base -> P1 -> P2:
- **Performance consistently decreases** across all models.
- **Variance consistently increases** across all models.
- **Rate of accuracy drop accelerates** with difficulty (not linear).

Selected results:
| Model | GSM-M1 | GSM-Symb | GSM-P1 | GSM-P2 |
|-------|--------|----------|--------|--------|
| Gemma2-9b-it | 84.4 (+/-2.4) | 79.1 (+/-3.0) | 68.1 (+/-4.8) | 41.8 (+/-6.0) |
| Phi-3.5-mini | 87.6 (+/-2.0) | 82.1 (+/-3.4) | 64.8 (+/-5.4) | 44.8 (+/-6.3) |
| GPT-4o | 94.4 (+/-1.6) | 94.9 (+/-1.9) | 93.9 (+/-2.6) | 88.0 (+/-3.4) |
| o1-mini | 94.9 (+/-1.5) | 94.5 (+/-1.6) | 94.3 (+/-2.6) | 89.1 (+/-3.6) |
| o1-preview | 93.6 (+/-1.7) | 92.7 (+/-1.8) | 95.4 (+/-1.7) | 94.0 (+/-2.4) |

The accelerating drop rate is consistent with pattern matching rather than formal reasoning: reasoning steps increase linearly but the probability of correct pattern matching decreases faster than linearly.

### 5.4 GSM-NoOp: Catastrophic Failures with Irrelevant Information (Section 4.4)

**The most striking finding.** Adding a single irrelevant but seemingly relevant clause causes dramatic performance drops.

**Performance drops (GSM8K -> GSM-NoOp):**
| Model | Drop |
|-------|------|
| Phi-3-mini-128k-instruct | **-65.7%** |
| Phi-3-small-128k-instruct | -64.0% |
| Gemma2-9b / Gemma2-9b-it | -63.0% |
| Phi-3.5-mini-instruct | -62.5% |
| Gemma2-27b-it | -59.7% |
| Llama3-8b-instruct | -57.4% |
| GPT-4o-mini | -40.0% |
| GPT-4o | -32.0% |
| o1-mini | -29.1% |
| o1-preview | **-17.5%** |

**Common failure modes:**
- Models blindly convert statements to operations (e.g., "5 were smaller" -> subtract 5).
- Models interpret "discount" as "multiply" regardless of context.
- Models apply irrelevant inflation rates (e.g., "prices were 10% cheaper last year" applied to current prices).
- Models subtract donated items from price comparisons where donation is irrelevant to cost difference.

**Few-shot remediation fails:**
- **NoOp-Symb** (8 shots of the same question from GSM-Symbolic): Performance remains within standard deviation of the No-Op baseline. Even with 8 demonstrations showing the correct reasoning chain, the model still falls for the irrelevant clause.
- **NoOp-NoOp** (8 shots from different GSM-NoOp questions): Performance does not improve (Llama3-8b) or slightly decreases (Phi-3).
- **Notable exception:** Some weaker models (Gemma2b, Mistral-7b-v0.1) actually perform better on NoOp-Symb despite being much worse on GSM8K/GSM-Symbolic overall -- a surprising finding.

---

## 6. The "Pattern Matching vs. Genuine Reasoning" Argument

### Central Thesis
Current LLMs perform **probabilistic pattern matching** rather than formal logical reasoning. They search for and replicate reasoning steps similar to those seen in training data, without genuine understanding of mathematical concepts.

### Evidence Supporting This Thesis

1. **Variance under irrelevant changes:** If models truly reasoned, changing names or swapping equivalent numerical values should not cause performance variance. Yet all models show significant variance.

2. **Sensitivity to numerical values but not names:** Models are more sensitive to number changes than name changes, suggesting they have memorized or pattern-matched specific numerical configurations from training.

3. **Accelerating performance decline with difficulty:** If models performed formal reasoning, adding one reasoning step should cause a roughly constant performance decrease. Instead, the rate of drop accelerates, consistent with exponentially decreasing probability of correct pattern matching as the number of required tokens increases (per Schaeffer et al., 2023).

4. **Catastrophic failure on NoOp:** Models blindly convert textual cues (e.g., "smaller," "discount," "inflation," "donate") into mathematical operations without understanding whether those operations are logically warranted. This is exactly what pattern matching would produce: the training data consistently pairs words like "smaller" with subtraction, so the model applies subtraction regardless of context.

5. **Few-shot demonstrations don't help:** Even when given 8 examples of the exact same question (with correct reasoning chains that ignore the irrelevant clause), models still fail on the NoOp variant. This suggests the issue is deeper than in-context learning can address.

6. **Fine-tuning on easier tasks doesn't generalize:** Fine-tuning Phi-3.5 on GSM-P1 data slightly improves P1 performance but does not improve (and may decrease) P2 performance, suggesting models are not learning generalizable reasoning skills.

### Supporting Literature Cited
- **Jiang et al. (2024):** Strong token bias; reasoning output changes when a single input token changes.
- **Li et al. (2024b):** Single transformer layer learns one-nearest-neighbor, explaining sensitivity to input tokens.
- **Schaeffer et al. (2023):** Probability of correct multi-token answers decreases exponentially with number of tokens.
- **Dziri et al. (2023):** Full computation subgraphs appear more frequently in training data for correct predictions.
- **Razeghi et al. (2022):** Correlation between pretraining term frequency and test performance.
- **Kambhampati (2024), Valmeekam et al. (2022, 2024):** LLMs cannot plan; reasoning is not formal.

---

## 7. Implications for Evaluating Mathematical Reasoning

1. **Single-point metrics on static benchmarks are unreliable.** GSM8K scores should be viewed as a distribution, not a point estimate. The reported "accuracy" can be misleading.

2. **Data contamination is likely widespread.** The systematic right-shift of GSM8K performance relative to GSM-Symbolic distributions (21/25 models) suggests contamination.

3. **Evaluation should use multiple instantiations.** Performance should be reported as mean +/- standard deviation across many variants of the same questions.

4. **Robustness to irrelevant information must be tested.** GSM-NoOp reveals a critical flaw that standard benchmarks completely miss.

5. **Difficulty scaling matters.** Models that appear strong on base-difficulty questions may collapse on slightly harder variants. GSM-Symbolic's difficulty levels (M1, base, P1, P2) provide a more complete picture.

6. **Even "reasoning-focused" models (o1-series) are not immune.** o1-preview shows the strongest results overall but still drops 17.5% on GSM-NoOp and exhibits performance variance.

7. **The limitations are more pronounced for harder benchmarks.** GSM8K/GSM-Symbolic questions require only basic arithmetic (addition, subtraction, multiplication, division). The fragility observed here will likely be worse on more challenging mathematical tasks.

---

## 8. Dataset Availability and Format

- **GSM-Symbolic** is constructed from **symbolic templates** that allow generating unlimited question variants.
- Each template includes: question text with placeholders, variable definitions with domains, conditions for validity, and a parameterized solution.
- The benchmark uses **100 templates** with **50 samples each** = 5,000 examples per variant.
- **Variants available:** GSM-Symbolic (base), GSM-M1, GSM-P1, GSM-P2, GSM-NoOp.
- The paper does not explicitly state a public release URL, but the templates and generation methodology are described in detail.
- Related datasets for comparison: iGSM (Ye et al., 2024), GSM-IC (Shi et al., 2023), GSM-Plus (Li et al., 2024a), GSM1K (Zhang et al., 2024, not publicly available).

---

## 9. Full Results Table (Appendix A.2, Table 1)

Complete 8-shot results for all models across all GSM-Symbolic variants:

| Model | GSM8K (Full) | GSM8K (100) | Symbolic-M1 | Symbolic | Symbolic-P1 | Symbolic-P2 | Symbolic-NoOp |
|-------|-------------|-------------|-------------|----------|-------------|-------------|---------------|
| Gemma2b | 12.1 | 11.0 | 24.5 (+/-3.85) | 8.2 (+/-2.21) | 3.6 (+/-2.13) | 1.5 (+/-1.63) | 4.7 (+/-1.99) |
| Gemma2-9b-it | 85.3 | 87.0 | 84.4 (+/-2.36) | 79.1 (+/-2.99) | 68.1 (+/-4.77) | 41.8 (+/-6.00) | 22.3 (+/-5.11) |
| Gemma2-27b-it | 89.7 | 92.0 | 90.2 (+/-1.86) | 88.3 (+/-2.56) | 80.7 (+/-4.07) | 63.4 (+/-4.14) | 30.0 (+/-3.39) |
| Phi-3-mini | 83.7 | 85.0 | 85.9 (+/-2.44) | 80.7 (+/-2.94) | 63.4 (+/-5.63) | 37.5 (+/-5.76) | 18.0 (+/-3.83) |
| Phi-3-medium | 87.3 | 89.0 | 89.6 (+/-1.65) | 82.5 (+/-2.86) | 75.8 (+/-3.89) | 53.1 (+/-4.80) | 29.4 (+/-4.18) |
| Phi-3.5-mini | 84.9 | 88.0 | 87.6 (+/-1.98) | 82.1 (+/-3.38) | 64.8 (+/-5.43) | 44.8 (+/-6.32) | 22.4 (+/-4.03) |
| Mathstral-7b | 80.1 | 80.0 | 82.9 (+/-2.87) | 74.0 (+/-3.49) | 57.4 (+/-5.20) | 35.5 (+/-5.07) | 20.4 (+/-3.58) |
| Llama3-8b-it | 76.0 | 74.0 | 79.5 (+/-3.62) | 74.6 (+/-2.94) | 53.8 (+/-4.54) | 28.3 (+/-4.37) | 18.6 (+/-3.86) |
| GPT-4o-mini | 94.2 | 95.0 | 92.5 (+/-1.63) | 91.7 (+/-2.02) | 81.1 (+/-3.05) | 72.4 (+/-4.57) | 54.1 (+/-3.85) |
| GPT-4o | 95.2 | 95.0 | 94.4 (+/-1.62) | 94.9 (+/-1.87) | 93.9 (+/-2.59) | 88.0 (+/-3.43) | 63.1 (+/-4.53) |
| o1-mini | 95.1 | 93.0 | 94.9 (+/-1.49) | 94.5 (+/-1.58) | 94.3 (+/-2.57) | 89.1 (+/-3.56) | 66.0 (+/-4.60) |
| o1-preview | 94.9 | 96.0 | 93.6 (+/-1.68) | 92.7 (+/-1.82) | 95.4 (+/-1.72) | 94.0 (+/-2.38) | 77.4 (+/-3.84) |

---

## 10. Key Takeaways for Our Research

1. **Symbolic templates are a powerful evaluation methodology.** They enable measuring performance as a distribution rather than a point, revealing fragility hidden by static benchmarks.

2. **Numerical sensitivity is the primary vulnerability.** Models are far more affected by number changes than name changes, suggesting they have memorized numerical patterns from training.

3. **Irrelevant information is devastating.** The GSM-NoOp results (up to 65% drops) reveal that models convert textual cues to operations without understanding, the clearest evidence against genuine reasoning.

4. **Difficulty scaling reveals exponential fragility.** The accelerating performance decline with added clauses is consistent with pattern matching (each additional step compounds error probability) rather than reasoning (which would show linear degradation at worst).

5. **Few-shot and fine-tuning don't fix the core problem.** Neither providing demonstrations of correct reasoning nor fine-tuning on easier variants improves performance on harder or NoOp variants, suggesting the limitation is architectural/fundamental rather than data-related.

6. **Even the strongest models (o1-preview) are affected.** While o1-preview maintains high performance on difficulty scaling, it still drops 17.5% on NoOp and makes clear conceptual errors (e.g., applying irrelevant inflation rates).

7. **Relevance to RLVR:** This paper provides strong motivation for training approaches (like RLVR) that could push models toward more robust reasoning rather than pattern matching. The GSM-Symbolic methodology could be used to evaluate whether RLVR-trained models show less variance and better NoOp robustness.
