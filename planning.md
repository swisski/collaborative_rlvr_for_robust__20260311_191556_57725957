# Research Plan: Collaborative RLVR for Robust Reasoning

## Motivation & Novelty Assessment

### Why This Research Matters
RLVR-trained LLMs achieve impressive math benchmark scores but are brittle — they exploit dataset patterns, produce unfaithful reasoning chains, and fail under simple rephrasings (GSM-Symbolic shows up to 65% accuracy drops). If collaborative reasoning (multi-agent debate) can improve robustness at inference time, this has immediate practical value: it provides a training-free method to make existing RLVR models more reliable without retraining.

### Gap in Existing Work
From the literature review, SDRL (Liu et al., 2026) demonstrates that collaborative RLVR training improves both solo and debate performance on MATH-500 and AIME, but **never evaluates on robustness benchmarks** like GSM-Symbolic. Meanwhile, GSM-Symbolic (Mirzadeh et al., 2024) reveals severe brittleness but doesn't test collaborative approaches as a mitigation. No prior work connects multi-agent debate to robustness evaluation.

### Our Novel Contribution
We test whether collaborative reasoning (multi-agent debate) specifically improves **robustness** of mathematical reasoning, measured by performance stability across symbolic variants (GSM-Symbolic). This bridges the gap between the debate literature (which shows accuracy gains) and the robustness literature (which identifies the brittleness problem).

### Experiment Justification
- **Experiment 1 (Single Agent Baseline):** Establishes base accuracy and robustness levels on GSM8K vs GSM-Symbolic.
- **Experiment 2 (Self-Consistency):** Tests whether simple sampling diversity (majority voting) improves robustness, serving as a non-collaborative baseline.
- **Experiment 3 (Collaborative Debate):** Tests the core hypothesis — does forcing models to externalize, challenge, and defend reasoning improve robustness beyond what sampling diversity alone provides?
- **Experiment 4 (Heterogeneous Debate):** Tests whether using models with different inductive biases (different model families) further improves robustness.

---

## Research Question
Does collaborative reasoning (multi-agent debate) improve the **robustness** of LLM mathematical reasoning, as measured by reduced performance degradation on symbolic variants (GSM-Symbolic) compared to original problems (GSM8K)?

## Background and Motivation
RLVR produces strong but brittle math reasoners. Multi-agent debate improves accuracy. But does debate specifically improve *robustness*? The hypothesis is that when a solution must convince another agent, shallow heuristics break while genuine reasoning survives — making debate particularly effective on distribution-shifted problems.

## Hypothesis Decomposition
- **H1:** Multi-agent debate improves absolute accuracy on GSM8K (replicating prior work).
- **H2:** Multi-agent debate improves absolute accuracy on GSM-Symbolic.
- **H3 (Core):** The *robustness ratio* (GSM-Symbolic accuracy / GSM8K accuracy) is higher for debate than for single-agent or self-consistency approaches. That is, debate disproportionately helps on distribution-shifted problems.
- **H4:** Heterogeneous debate (different model families) yields higher robustness than homogeneous debate.

## Proposed Methodology

### Approach
Inference-time study comparing single-agent, self-consistency, and debate approaches on both standard (GSM8K) and robustness (GSM-Symbolic) benchmarks using small math-capable models on CPU.

**Why inference-time (not training)?** Full RLVR training requires multi-GPU clusters. However, the core hypothesis — that collaborative reasoning improves robustness — can be tested at inference time. If debate doesn't improve robustness at inference time, training for it is unlikely to help. If it does, this motivates future training-time work (collaborative RLVR).

### Models
- **Primary:** Qwen2.5-Math-1.5B-Instruct (math-specialized, strong for size)
- **Secondary (for heterogeneous debate):** A different small model (e.g., Phi-3-mini or SmolLM2) to test whether different inductive biases improve debate robustness

### Experimental Steps

#### Step 1: Data Preparation
- Select 50 GSM8K test problems
- For each, identify corresponding GSM-Symbolic variants (same original_id)
- This gives paired data: same problem structure, different numbers/names

#### Step 2: Single Agent Baseline
- Greedy decoding (temperature=0)
- Record: answer, correctness, solution text
- Run on both GSM8K and GSM-Symbolic

#### Step 3: Self-Consistency (SC@3)
- 3 samples per problem at temperature=0.7
- Majority vote on extracted answers
- This controls for sampling diversity without collaboration

#### Step 4: Collaborative Debate (1 round)
- Two independent solutions (temperature=0.7)
- Each "agent" sees the other's solution and generates a revised answer
- Final answer: majority vote across all responses (initial + revised)
- This tests whether the critique/revision process specifically helps

#### Step 5: Heterogeneous Debate
- Same protocol as Step 4 but using two different models
- Tests whether different inductive biases improve debate effectiveness

### Baselines
1. **Single agent (greedy):** Lower bound
2. **Self-consistency (SC@3):** Controls for sampling diversity
3. **Random baseline:** Expected accuracy from random guessing

### Evaluation Metrics
1. **Accuracy:** % correct on each benchmark
2. **Robustness Ratio (RR):** accuracy_GSM_Symbolic / accuracy_GSM8K (1.0 = perfectly robust)
3. **Debate Delta (Δ):** accuracy_debate - accuracy_single
4. **Robustness Delta:** RR_debate - RR_single (positive = debate helps robustness)
5. **Variance across symbolic variants:** For problems with multiple variants, measure accuracy variance

### Statistical Analysis Plan
- McNemar's test for paired accuracy comparisons (same problems, different methods)
- Bootstrap confidence intervals for accuracy and robustness ratios
- Significance level: α = 0.05
- Effect size: Cohen's h for proportion differences

## Expected Outcomes
- **If H3 supported:** Debate shows higher robustness ratio than single-agent/SC, suggesting collaborative reasoning specifically helps with distribution shifts.
- **If H3 refuted:** Debate improves accuracy uniformly (or not at all), suggesting robustness gains require training-time changes (collaborative RLVR).

## Timeline and Milestones
1. Environment setup & model download: ~15 min
2. Data preparation: ~10 min
3. Implementation: ~30 min
4. Run experiments (CPU inference): ~60-90 min
5. Analysis & visualization: ~20 min
6. Documentation: ~20 min

## Potential Challenges
- **CPU inference speed:** Mitigated by using 1.5B model, batching, and limiting to 50 problems
- **Small model capability:** 1.5B models have limited math ability (~50-70% on GSM8K). This is acceptable because we measure *relative* robustness, not absolute performance.
- **Small sample size:** 50 problems limits statistical power. Mitigated by using paired tests and bootstrap CIs.

## Success Criteria
The research succeeds if we can:
1. Establish clear accuracy numbers for all conditions on both benchmarks
2. Compute robustness ratios with confidence intervals
3. Determine whether debate specifically helps robustness (H3)
4. Document findings rigorously regardless of direction
