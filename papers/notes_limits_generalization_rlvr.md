# Notes: Limits of Generalization in RLVR: Two Case Studies in Mathematical Reasoning

**Paper:** Alam & Rastogi (2025), arXiv:2510.27044v2, NeurIPS 2025 MATH-AI Workshop
**Code:** https://github.com/xashru/rlvr-seq-generalization

---

## 1. Core Thesis

RLVR (Reinforcement Learning with Verifiable Rewards) improves evaluation metrics on mathematical reasoning tasks, but these improvements often arise from **reinforcing superficial heuristics rather than acquiring genuine new reasoning strategies**. The paper provides rigorous evidence for this claim through two fully verifiable combinatorial problems.

---

## 2. Experimental Design

### 2.1 Task Selection Rationale

The authors deliberately chose tasks with **fully verifiable, unique optimal solutions** so that correctness of both the final answer AND the intermediate reasoning process can be precisely evaluated. This addresses a key gap: most prior RLVR studies rely on benchmarks where reasoning correctness is difficult to verify.

### 2.2 Tasks

**Activity Scheduling:**
- Given activities with start/finish times, select the maximum non-overlapping subset.
- Solvable by greedy earliest-finish-time algorithm.
- Each instance is constructed to have a unique optimal solution (verified via DP counting).
- Ground truth: the ID sequence sorted by finish time, plus the count.

**Longest Increasing Subsequence (LIS):**
- Given a sequence of integers, find the longest strictly increasing subsequence.
- Solvable by dynamic programming (O(n^2) counting ensures uniqueness).
- Reconstruction via patience sorting with predecessor links.
- Each instance has a single unique optimal LIS.

### 2.3 Dataset Construction

- 2000 instances per task, half with algorithmic hints, half without.
- Sequence lengths 5-16.
- **Disjoint length ranges for train and test** to avoid leakage (462 test for Activity, 428 for LIS).
- Uniqueness of the optimal solution is enforced algorithmically for every instance.

### 2.4 Model and Training

- **Base model:** Qwen2.5-7B-Instruct (with additional Llama-3.1-8B results in appendix).
- **Algorithm:** GRPO (Group Relative Policy Optimization) via the verl framework.
- 256 prompts per PPO update, 8 rollouts, 20 epochs (120 updates), learning rate 1e-6, no KL penalty.
- Max generation length: 2048 tokens (7680 for LIS with format reward).

### 2.5 Reward Functions (5 Designs Tested)

| Reward | Description | Scope |
|--------|-------------|-------|
| r_ans (Answer-only) | Binary: 1 iff predicted count matches ground truth | Both tasks |
| r_ans+fmt (Answer + Format) | 0.9 * answer correctness + 0.1 * format compliance (think tags + valid output) | LIS only |
| r_ids,exa (Exact-IDs) | Binary: 1 iff predicted ID sequence exactly matches optimum | Both tasks |
| r_ids,pre (Prefix-IDs) | Partial credit proportional to longest correct prefix, with length penalty | Both tasks |
| r_sort (Sorting-Match) | Binary: 1 iff extracted sorted sequence matches canonical sort order | Activity only |

### 2.6 Evaluation Protocol

- 256 samples per instance (temperature 0.6, top-p 0.95).
- **Two accuracy notions:** Acc_ans (answer count correct) and Acc_ids (exact ID sequence match).
- **Two aggregation methods:**
  - **Pass@k:** At least one of k samples correct.
  - **Self-consistency (SC):** Majority vote over k samples matches ground truth.
- Full curves plotted for k = 1 to 256.

---

## 3. Specific Generalization Failures Found

### 3.1 Failure 1: LIS Answer-Only Reward Collapses Reasoning

- Training LIS with r_ans causes **rapid collapse of chain-of-thought reasoning**: after just a few PPO updates, the model drops intermediate reasoning and outputs terse final answers.
- Mean response length plummets (from ~500 tokens to near 0).
- Entropy drops sharply, mirroring the response length collapse.
- The model achieves higher SC on answer accuracy but **fails on exact sequence ID evaluation**, with low Pass@k and SC compared to the base model.

### 3.2 Failure 2: Activity Scheduling Sorting Is Superficial

- Models frequently emit a "sorted preface" (a sorted version of input activities) as the first step, mimicking the greedy algorithm's sorting step.
- However, **exact sorting accuracy remains extremely low (~2%)** even for models with high overall sequence correctness (Pass@256_ids ~ 0.60).
- The sorting step is not reliably driving the final schedule -- it is a **superficial surface pattern** that neither matches the canonical order nor drives the underlying decision rule.

### 3.3 Failure 3: Sorting Reward Causes Catastrophic Collapse

- Training with r_sort alone led to **catastrophic failure**: both Acc_ans and Acc_ids fell to nearly 0%.
- The model learned to output the sorted sequence itself as the final answer, without applying the non-overlap constraint.
- Combined objective (equal weights on r_ans + r_ids + r_sort) restored performance but did NOT improve sorting accuracy -- learning to sort in isolation confers no benefit.

### 3.4 Failure 4: Curriculum Learning with Sorting Hinders Recovery

- Training with r_sort for 10 PPO steps then switching to r_ans allowed recovery.
- But training with r_sort for 20 or 30 steps **severely hindered recovery** -- after 30 steps of r_sort, the model failed to improve Acc_ans even on training data.
- This shows RLVR can become stuck in degenerate strategies that are hard to escape.

### 3.5 Failure 5: LIS Improvements Are Heuristic, Not Algorithmic

- Random Forest regression of model outputs against input features achieves R^2 = 0.74-0.87 for RLVR models vs. R^2 = -0.002 for the base model.
- This means RLVR-trained outputs are **highly predictable from simple statistical features** of the input (global scale, order structure, run patterns, simple heuristic approximations).
- RLVR amplifies systematic heuristics aligned with task structure, without learning the actual DP algorithm.

### 3.6 Failure 6: Hints Do Not Help

- No significant performance difference between hinted and unhinted prompts across both tasks and all model variants.
- Models do not substantially benefit from algorithmic guidance provided in prompts, suggesting they are not following the prescribed reasoning strategy.

---

## 4. How Models Exploit Patterns Rather Than Learn Robust Reasoning

### 4.1 Surface-Level Pattern Matching

- **Activity Scheduling:** Models learn to emit a "sorted preface" that looks like the first step of the greedy algorithm but does not actually implement correct sorting. The sorting step is decorative rather than functional.
- **LIS:** Models converge to heuristic shortcuts (captured by simple statistical features) rather than implementing dynamic programming. With r_ans+fmt, nearly 100% of responses contain Python code (vs. 35.1% for base), but the model does not execute the code step-by-step to derive the answer -- the code is superficial structure.

### 4.2 Heuristic Feature Exploitation (LIS Regression Analysis)

Features that predict RLVR model outputs include:
- **Global scale:** sequence length, range, dispersion, quantiles
- **Order structure:** increase ratios, inversion ratio, sign changes
- **Run structure:** longest increasing/decreasing runs, monotone counts, local extrema, record highs/lows
- **Simple LIS heuristics:** greedy length, beam-limited LIS, limited backtracking
- **Patience-sorting tails:** mean, std, IQR, slope of tails vector

The high R^2 values (0.74-0.87) demonstrate that RLVR models are essentially implementing **weighted combinations of these simple heuristics** rather than the actual LIS algorithm.

### 4.3 Reward Hacking via Format

- Under r_ans on LIS, the model discovers it can maximize reward by dropping chain-of-thought entirely and just guessing the numeric answer (a small integer).
- Under r_ans+fmt, the model preserves formatting but fills it with superficial content (e.g., Python code that is not actually executed).
- Both strategies achieve similar Acc_ans and Acc_ids despite radically different generation styles, confirming reliance on comparable heuristics.

---

## 5. Distribution Shift Experiments and Results

### 5.1 Train-Test Split by Length

- Train and test use **disjoint length ranges** (sequence lengths 5-16 split into non-overlapping subsets).
- This creates a natural distribution shift: models must generalize to unseen problem sizes.
- RLVR shows improvements on the test set, but the improvements are attributable to heuristic amplification rather than algorithmic generalization.

### 5.2 Key Quantitative Results

**Activity Scheduling (answer-only reward r_ans):**
| Metric | Base | RLVR |
|--------|------|------|
| Pass@256_ans | ~1.0 | Slightly lower |
| SC@256_ans | 0.24 | 0.68 |
| Pass@256_ids | 0.14 | 0.64 |
| SC@256_ids | 0.004 | 0.34 |

**Activity Scheduling (sequence rewards r_ids,exa / r_ids,pre):**
| Metric | r_ans | r_ids,exa | r_ids,pre |
|--------|-------|-----------|-----------|
| SC@256_ids | 0.34 | 0.72 | 0.71 |
| SC@256_ans | 0.68 | 0.74 | 0.75 |

**LIS (various rewards):**
| Metric | Base | r_ans | r_ans+fmt | r_ids,exa | r_ids,pre |
|--------|------|-------|-----------|-----------|-----------|
| SC@256_ans | ~0.24-0.31 | ~0.58 | ~0.62 | ~0.63 | ~0.70 |
| SC@256_ids | ~0.05 | ~0.08 | ~0.08 | ~0.42 | ~0.40 |

### 5.3 Sorting Quality Across Models

| Model | Exact Sorting Accuracy | Mean LCS |
|-------|----------------------|----------|
| Base | 0.17% | 0.248 |
| RL(r_ans) | 0.43% | 0.290 |
| RL(r_ids,exa) | 2.01% | 0.517 |
| RL(r_ids,pre) | 1.85% | 0.444 |

The gap between ~2% exact sorting and ~60% sequence correctness is a key indicator that the model is NOT following the greedy algorithm faithfully.

### 5.4 Heuristic Predictability (Random Forest R^2)

| Model | R^2_test | MAE_test |
|-------|----------|----------|
| Base | -0.002 | 2.745 |
| r_ans | 0.741 | 0.269 |
| r_ids,exa | 0.781 | 0.289 |
| r_ids,pre | 0.841 | 0.227 |
| r_ans+fmt | 0.867 | 0.209 |

---

## 6. Mathematical Reasoning Benchmarks Used

The paper uses **custom-constructed benchmarks** rather than standard ones, precisely to enable rigorous verification:

1. **Activity Scheduling** -- unweighted interval scheduling with unique greedy optimum (462 test instances).
2. **Longest Increasing Subsequence (LIS)** -- strict LIS with unique DP-verified optimum (428 test instances).

Both benchmarks feature:
- Unique ground-truth solutions (verified algorithmically).
- Structured output format (\ids{...} and \answer{...}) enabling both answer-level and sequence-level evaluation.
- Disjoint train/test length ranges to test generalization.
- Half with algorithmic hints, half without.

The paper also validates findings on **Llama-3.1-8B** (Appendix F), showing that relative trends between base and RLVR-trained variants mirror those observed with Qwen.

---

## 7. Key Findings About Brittleness

### Finding 1: RLVR Can Yield Apparent Generalization Without Reasoning Gains
RLVR improves answer-level metrics (Acc_ans) on both tasks, but only Activity Scheduling shows genuine reasoning improvement (Acc_ids). On LIS, improvements come from superficial heuristics or formatting strategies.

### Finding 2: Answer-Only Pass@k Overstates Capability
Standard Pass@k on small-integer answers saturates quickly and masks the absence of genuine reasoning. Stricter metrics like Acc_ids and self-consistency provide more faithful measures.

### Finding 3: Chain-of-Thought Collapse Under Answer-Only Reward
RLVR with answer-only reward on LIS causes the model to abandon intermediate reasoning entirely, converging to terse outputs that guess the numeric answer.

### Finding 4: Sorting Preface Is Decorative, Not Functional
Despite frequently emitting sorting steps, models achieve only ~2% exact sorting accuracy. The surface behavior mimics the algorithm without implementing it.

### Finding 5: Sub-Goal Rewards Can Be Counterproductive
Training on sorting reward alone causes catastrophic collapse. Even curriculum approaches with extended sorting pre-training hinder subsequent task learning.

### Finding 6: RLVR Amplifies Heuristics, Not Algorithms
RLVR-trained model outputs are highly predictable from simple statistical features (R^2 up to 0.87), showing that RLVR amplifies systematic heuristics rather than inducing genuine algorithmic reasoning.

### Finding 7: Hints Have No Effect
Algorithmic hints in prompts produce no measurable benefit, suggesting models are not following the prescribed reasoning strategy regardless of guidance.

### Finding 8: Base Model Ability Acts as a Ceiling
Consistent with prior work, RLVR gains largely reflect re-weighting and stabilizing existing solutions rather than expanding the model's reasoning repertoire. Reasoning coverage can contract at larger sample sizes.

---

## 8. Implications for RLVR Training Design

### 8.1 Reward Design

- **Answer-only rewards are insufficient** -- they incentivize shortcut exploitation and can collapse chain-of-thought reasoning.
- **Sequence-aware rewards** (exact-IDs, prefix-IDs) provide meaningful improvement in sequence fidelity and should be preferred when verifiable intermediate steps exist.
- **Sub-goal rewards must be combined carefully** -- isolated sub-goal rewards (e.g., sorting-only) can cause catastrophic collapse by rewarding partial behaviors that do not compose into valid solutions.
- **Format rewards** help prevent reasoning collapse but do not ensure genuine reasoning -- they can encourage structured-but-superficial outputs (e.g., non-executed code).

### 8.2 Evaluation Design

- **Standard Pass@k on answer accuracy is misleading**, especially for small-integer answers. It over-credits lucky completions.
- **Sequence-level evaluation (Acc_ids)** and **self-consistency** are more faithful measures of genuine reasoning.
- **Benchmarks should have unique, verifiable solutions** to enable precise measurement of both answer and process correctness.
- **Disjoint train/test distributions** (e.g., by problem size) are essential for detecting heuristic overfitting.

### 8.3 Structural Concerns

- RLVR appears to **stabilize existing competencies** rather than induce new reasoning strategies -- the base model's capability acts as a ceiling.
- Gains from RLVR largely reflect **re-weighting of existing solution distributions**, not discovery of new solution strategies.
- Even when models produce outputs that superficially resemble correct algorithms (e.g., sorting steps), the intermediate steps may be **unfaithful to the actual decision process** (consistent with Turpin et al. 2023 on unfaithful chain-of-thought).

---

## 9. Recommendations for Improving Robustness

### From the Paper

1. **Use benchmarks with fully verifiable solutions** that enable measuring both answer accuracy and reasoning process fidelity.
2. **Employ sequence-level metrics** (Acc_ids, self-consistency) rather than relying solely on answer-level Pass@k.
3. **Design datasets that disentangle genuine reasoning from shortcut exploitation** -- e.g., unique optima, disjoint train/test distributions.
4. **Incorporate intermediate/auxiliary objectives** (sequence-aware rewards) to improve reasoning fidelity, but combine them carefully with task-level rewards.
5. **Apply mechanistic interpretability** to probe whether RLVR improvements reflect genuine task learning or superficial strategies.
6. **Test across multiple model families and scales** -- the paper found consistent trends across Qwen and Llama, but shortcut behaviors can be model-dependent.

### Inferred from Results

7. **Avoid reward designs that can be satisfied without reasoning** -- if the answer space is small (e.g., integers 1-16), answer-only rewards are especially vulnerable to shortcut exploitation.
8. **Monitor response length and entropy during training** as early indicators of reasoning collapse.
9. **Be cautious with curriculum approaches** -- extended training on sub-goal rewards can create degenerate attractors that are difficult to escape.
10. **Develop evaluation probes** (e.g., Random Forest regression on input features) to detect whether models are using heuristic shortcuts rather than algorithms.
11. **Consider process reward models** or verification of intermediate reasoning steps rather than relying solely on outcome-based rewards.

---

## 10. Connections to Prior Work

- **Yue et al. (2025):** RLVR does not incentivize reasoning beyond the base model's capability -- consistent with the ceiling effect observed here.
- **Wu et al. (2025):** "The invisible leash" -- RLVR may not escape the origin model's solution distribution.
- **Shao et al. (2025):** Spurious rewards can drive improvements in models with strong procedural biases -- consistent with the Qwen family generating structured-but-superficial code.
- **Wen et al. (2025):** Answer-only Pass@k can overstate capability; CoT-Pass@k is more reliable.
- **Wang et al. (2025):** A single training example can rival large-scale RLVR training, further questioning whether RLVR induces new reasoning.
- **Turpin et al. (2023):** LLMs produce unfaithful chain-of-thought explanations -- directly relevant to the finding that sorting prefaces are decorative.

---

## 11. Limitations

- Only two tasks tested (Activity Scheduling and LIS).
- Primary results on a single base model (Qwen2.5-7B-Instruct), with partial validation on Llama-3.1-8B.
- Conclusions may not generalize across task types, model families, or scales.
- The paper does not test larger models or more complex problem domains.
- No mechanistic interpretability analysis -- the paper calls for this as future work.
