# Notes: "Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs"

**Paper:** arXiv:2506.14245v2 (2 Oct 2025)
**Authors:** Xumeng Wen, Zihan Liu, Shun Zheng, Shengyu Ye, Zhirong Wu, Yang Wang, Zhijian Xu, Xiao Liang, Junjie Li, Ziming Miao, Jiang Bian, Mao Yang (Microsoft Research Asia, Peking University, CUHK, UCLA)

---

## 1. Core Thesis

RLVR **implicitly incentivizes correct (faithful) reasoning** in base LLMs, even though rewards are based solely on answer correctness. This challenges the competing hypothesis (Yue et al., 2025) that RLVR merely improves sampling efficiency without genuinely enhancing reasoning capacity.

The paper's central claim: once LLMs have been pre-trained with strong knowledge and logic priors that distinguish correct from incorrect chain-of-thought (CoT) reasoning, the GRPO gradient will systematically increase the probability of generating correct CoTs.

---

## 2. How RLVR Incentivizes Correct (Faithful) Reasoning

### 2.1 The Competing Hypothesis (Yue et al., 2025)
- Post-RLVR models improve Pass@1 but fail to improve Pass@K compared to base models.
- Hypothesis: all correct reasoning paths already exist in the base model; RLVR merely adjusts sampling probabilities (improves sampling efficiency) at the cost of reducing overall reasoning capacity.

### 2.2 This Paper's Perspective
- RLVR **promotes correct reasoning paths** and **mitigates spurious guesses**.
- The mechanism works through the GRPO advantage function: correct CoTs receive positive advantages and incorrect CoTs receive negative advantages, even when rewards only verify final answers.
- Key insight: the "Logic Prior" from pre-training ensures that correct CoTs are more likely to produce correct answers than incorrect CoTs, creating a systematic gradient signal favoring correct reasoning.

### 2.3 Theoretical Framework (Theorem 1: GRPO Implicitly Incentivizes Correct Reasoning)

**Problem Setup:**
- For question prompt q, sample G responses Y = {y1, ..., yG} from policy pi_theta.
- Each response yi has a CoT ci and answer ai.
- Define correctness indicators: I_CoT(ci) for CoT correctness, I_Ans(ai) for answer correctness.
- Reward R(yi) = I_Ans(ai) (binary, answer-correctness only).
- Standard GRPO advantage: A_hat(yi) = (R(yi) - mu_Y) / sigma_Y.

**Critical "Logic Prior" Assumption:**
- P(I_Ans(ai)=1 | I_CoT(ci)=1) = alpha > P(I_Ans(ai)=1 | I_CoT(ci)=0) = beta
- Correct CoTs have higher probability of inducing correct answers than incorrect CoTs.
- This assumption relies on strong knowledge and logic priors established during pre-training.

**Theorem 1 Statement:**
For any prompt q satisfying the Logic Prior assumption:
- E[A_hat(yi) | I_CoT(ci) = 1] > 0 (correct CoTs get positive expected advantage)
- E[A_hat(yi) | I_CoT(ci) = 0] < 0 (incorrect CoTs get negative expected advantage)
- The GRPO policy gradient increases the probability of generating correct CoTs (p_c) monotonically.

**Proof Sketch:**
- Group-level expected reward: mu = p_c * alpha + (1 - p_c) * beta
- Expected advantage for correct CoT converges to: (1 - p_c)(alpha - beta) / sigma > 0
- Expected advantage for incorrect CoT converges to: -p_c(alpha - beta) / sigma < 0
- Since alpha > beta (Logic Prior) and sigma > 0, both inequalities hold.

**Driving Factor:**
- The gap alpha - beta > 0 amplifies the advantage difference.
- As training progresses: alpha increases (more sound reasoning), beta decreases (fewer spurious correlations), widening the gap and accelerating coherent reasoning.
- As p_c -> 1, the advantage for correct CoTs approaches 0, ensuring convergence.

**Ideal Case (alpha -> 1, beta -> 0):**
- E[A_hat | correct CoT] -> sqrt((1-p_c)/p_c)
- E[A_hat | incorrect CoT] -> -sqrt(p_c/(1-p_c))

---

## 3. Evidence For/Against Reasoning Faithfulness in RLVR Models

### 3.1 Evidence FOR (presented in this paper)

**Extended Reasoning Boundary (Math):**
- Using CoT-Pass@K (not just Pass@K), RLVR shows a consistent and significant performance gap over base LLMs across all K values (up to 1024) on AIME 2024 and AIME 2025.
- AIME 2025 gap is particularly pronounced (released after base model training cutoff, so no data contamination).
- Traditional Pass@K on math is unreliable because base LLMs can produce incorrect CoTs that coincidentally arrive at the ground truth (guessing).

**Extended Reasoning Boundary (Code):**
- AceReason-Nemotron-7B shows clear Pass@K improvements over DeepSeek-R1-Distill-Qwen-7B across multiple LiveCodeBench versions (v1-v6).
- Code verification relies on actual execution, significantly reducing guessing likelihood.
- Skywork-OR1-7B also shows consistent Pass@K gap over DeepSeek-R1-Distill-Qwen-7B on LiveCodeBench-v6.

**Training Dynamics Evidence:**
- P(CC|CA)^(q) (probability of correct CoT given correct answer) improves steadily during DAPO training (Figure 4).
- This improvement starts from the very beginning of training and generalizes to unseen test questions.
- CoT-Pass@K on AIME 2024/2025 improves for intermediate checkpoints (Step-30, Step-60, Step-210).

**SFT-Based Quality Evidence:**
- SFT on RLVR-generated CoTs produces models with progressively better generalization as RLVR training advances.
- SFT on DAPO CoTs nearly replicates the Pass@1 performance of DAPO-Qwen-32B.
- Even "incorrect" CoTs from later RLVR stages show improved overall quality.
- SFT on base LLM CoTs (filtered for correct answers) begins to mitigate guessing, confirming the value of RLVR-incentivized CoTs.

### 3.2 Evidence AGAINST / Limitations

**Pass@K Observations That Support Skeptics:**
- On standard Pass@K (answer-only), base LLMs catch up with and surpass post-RLVR models as K increases -- this is reproducible and aligns with Yue et al. (2025).
- On easier benchmarks (MATH-500, AMC23), RLVR effects are less pronounced; base LLM already solves problems correctly.
- On Minerva (physics problems), no improvement -- train-test domain mismatch (DAPO trained on math with integer answers).

**Failure Modes:**
- The Logic Prior assumption may not always hold -- base LLMs retain inherent biases and possibly fatal knowledge errors.
- Improper model biases can be unintentionally reinforced (suspected root cause of R1-Zero's poor readability and language mixing).
- After 400 DAPO steps, most training questions reach P(CA)^(q) ~ 1.0, but median P(CC|CA)^(q) is only ~0.7, indicating persistent imperfect reasoning.
- For distilled LLMs in math domains, RLVR primarily improves Pass@1 without extending the reasoning boundary (Skywork-OR1-Math-7B vs DeepSeek-R1-Distill-Qwen-7B).

---

## 4. Methodology for Measuring Reasoning Correctness

### 4.1 Novel Metric: CoT-Pass@K
- Evaluates success only when **both** the final answer AND the intermediate reasoning CoT are correct.
- Computed per-prompt: CoT-Pass@K^(q) = 1 - C(G-D, K) / C(G, K), where D = sum of (I_CoT(ci) * I_Ans(ai)).
- Addresses the fundamental flaw of Pass@K: base LLMs can guess correct answers with wrong reasoning, especially for hard problems with simple integer answers.

### 4.2 LLM-as-a-CoT-Judge Paradigm
- Uses DeepSeek-R1-0528-Qwen3-8B as a specialized verifier for mathematical CoT correctness.
- **Multi-verification approach** with n=3 independent verification attempts per CoT:
  - **All-correct:** Must pass all 3 verifications (mitigates false positives; FP rate decays as p_fp^n)
  - **Majority-correct:** Must pass majority of verifications
  - **Any-correct:** Must pass at least 1 verification (mitigates false negatives; FN rate decays as p_fn^n)
- Results reported as a shaded band between any-correct and all-correct bounds.
- Manual inspection performed on borderline cases (Pass@K positive but CoT-Pass@K zero) to validate reliability.

### 4.3 Verifier Prompt Template
- Instructs the verifier to analyze solutions step-by-step, checking: computational accuracy, logical consistency, conceptual understanding, completeness.
- Error categories: Calculation Error, Logical Error, Conceptual Error, Omission/Incompleteness, Other.
- Output: boxed{yes} or boxed{no}.

### 4.4 Key Training Indicators
- P(CA)^(q) = C/G: probability of producing correct answers for prompt q.
- P(CC|CA)^(q) = D/C: probability of correct CoT given correct answer.
- Both tracked throughout training on per-prompt basis.

---

## 5. Mathematical Benchmarks and Datasets Used

### 5.1 Math Benchmarks (Evaluation)
| Benchmark | Description | Key Observations |
|-----------|-------------|------------------|
| AIME 2025 | Competition math (released after base model cutoff) | Strongest extended boundary signal; no data contamination |
| AIME 2024 | Competition math | Significant CoT-Pass@K gap |
| MATH-500 | Mixed math problems | RLVR effects less pronounced (problems too easy or in pre-training data) |
| AMC23 | AMC competition problems | Similar -- base LLM catches up |
| Minerva | Physics + free-form answers | No improvement (domain mismatch with DAPO training) |

### 5.2 Code Benchmarks (Evaluation)
| Benchmark | Versions | Key Observations |
|-----------|----------|------------------|
| LiveCodeBench | v1 through v6 | Clear Pass@K improvements for RLVR models; medium and hard problems drive differentiation |

### 5.3 Training Data
- **DAPO-Math-17k:** Curated set of 17k mathematical problems (from BytedTsinghua-SIA/DAPO-Math-17k on HuggingFace).
- Training questions divided into "easy" (at least one correct answer in 64 rollouts from base LLM) and "hard."

### 5.4 Data Sources
- AIME 2025: huggingface.co/datasets/opencompass/AIME2025
- AIME 2024: huggingface.co/datasets/HuggingFaceH4/aime_2024
- MATH-500: huggingface.co/datasets/HuggingFaceH4/MATH-500
- AMC23: huggingface.co/datasets/math-ai/amc23
- Minerva: huggingface.co/datasets/math-ai/minervamath

---

## 6. Key Results on Reasoning Quality vs Accuracy

### 6.1 Pass@K vs CoT-Pass@K Divergence
- **Pass@K (answer-only):** Base LLM catches up and surpasses post-RLVR model for large K on math tasks -- reproduces prior findings.
- **CoT-Pass@K (answer + reasoning):** Post-RLVR model maintains consistent advantage across all K on AIME 2024/2025 -- reveals the hidden reasoning improvement.
- This divergence demonstrates that base LLMs achieve high Pass@K partly through guessing (wrong CoT, right answer).

### 6.2 Training Dynamics
- P(CA)^(q) approaches 1.0 quickly for most training questions.
- P(CC|CA)^(q) improves more slowly but steadily throughout training (from ~0.3 initially to ~0.7 median after 400 steps).
- Correct reasoning is incentivized from the very beginning (within the first 20 training steps / few gradient updates).
- Generalization to unseen test questions is observed from early training.

### 6.3 SFT Replication
- SFT on DAPO CoT data can nearly replicate DAPO-Qwen-32B performance -- demonstrating that the CoTs themselves carry the reasoning improvement.
- CoT quality (measured by downstream SFT generalization) improves monotonically with RLVR training stage.
- Even CoTs with identifiable errors from later RLVR stages produce better SFT outcomes than base LLM CoTs.

---

## 7. Comparison of Different RL Algorithms

### 7.1 GRPO (Group Relative Policy Optimization)
- Central algorithm studied in the paper; used by DeepSeek-R1.
- Advantage computed as group-relative normalization: A_hat(yi) = (R(yi) - mu_Y) / sigma_Y.
- Requires a "learnable group" (sigma_Y > 0) -- fails when all responses in a group are correct or all incorrect.
- Paper provides formal proof that GRPO implicitly incentivizes correct reasoning under the Logic Prior assumption.

### 7.2 PPO (Proximal Policy Optimization)
- Referenced as the style of optimization in DAPO training (each training step = one round of PPO-style optimization with 16 gradient updates).
- DAPO recipe builds on PPO-style training within the VERL framework.

### 7.3 DAPO (Decoupled Alignment via Policy Optimization)
- Open-source recipe that successfully reproduced R1-Zero using Qwen2.5-32B + 17k math problems.
- The paper reproduced DAPO training and used it as the primary experimental platform.
- Achieved ~44% Pass@1 (vs reported >50%), in line with third-party reproductions.

### 7.4 Other Mentioned Approaches
- **DPO (Direct Preference Optimization):** Mentioned as a preference optimization framework for future RLVR algorithms.
- **TRPA (Trust Region Preference Approximation):** Referenced as potential new algorithm direction.
- **R1-Zero approach:** Pure RLVR from base model; challenges include poor readability and language mixing, explained by the paper as violations of the Logic Prior assumption (alpha not close to 1, beta not close to 0).
- **Distillation:** Shown to be effective because it directly teaches correct CoTs; for 32B base models where (p_c, alpha, beta) are in poor initial states, distillation outperforms pure RLVR.

---

## 8. Training Details and Hyperparameters

### 8.1 DAPO Reproduction Setup
- **Hardware:** 32 AMD MI300X GPUs
- **Framework:** VERL (HybridFlow)
- **Training Duration:** Over two weeks
- **Base Model:** Qwen2.5-32B
- **Training Data:** DAPO-Math-17k (17k math problems with integer answers)
- **Group Size (G):** 64 rollouts per prompt (used for both training and evaluation sampling)
- **Training Steps:** ~400 steps total
- **Per-step updates:** 16 gradient updates per training step (PPO-style)
- **Sampling:** Up to K=1024 for evaluation

### 8.2 CoT Verification Setup
- **Verifier Model:** DeepSeek-R1-0528-Qwen3-8B
- **Number of verification attempts:** n = 3 per CoT
- **Aggregation strategies:** any-correct, all-correct, majority-correct

### 8.3 Evaluation Sampling
- **Sampling Number (K):** Evaluated across 2^0 to 2^10 (1 to 1024) for Pass@K and CoT-Pass@K curves.
- **Temperature/sampling details:** Not explicitly stated beyond using the DAPO prompt template.

---

## 9. Implications for Designing RLVR Systems

### 9.1 The Logic Prior is Critical
- RLVR effectiveness depends on the base LLM having established strong knowledge and logic priors from pre-training.
- If the Logic Prior assumption is violated (alpha not >> beta), RLVR may reinforce incorrect reasoning patterns.
- This explains why R1-Zero works poorly for 32B models but well for larger models with better pre-training.

### 9.2 Benchmark Selection Matters
- Easy benchmarks (MATH-500, AMC23) may not reveal true RLVR improvements.
- Benchmarks must be: (a) challenging enough, (b) free from data contamination, (c) domain-matched with training data.
- AIME 2025 (post-cutoff) and LiveCodeBench (execution-verified, evolving) are ideal.
- Call for **live, evolving benchmarks** to avoid contamination risks.

### 9.3 Evaluation Metrics Must Account for Reasoning
- **Pass@K alone is insufficient** -- it cannot distinguish correct reasoning from lucky guessing.
- **CoT-Pass@K** is proposed as a more reliable metric for math reasoning.
- For code tasks, execution-based verification naturally prevents guessing, making Pass@K more reliable.
- Developing **lightweight yet powerful CoT verifiers** is a pressing need.

### 9.4 Training Curriculum Considerations
- RLVR begins to incentivize correct reasoning from the very first training steps.
- Once P(CA)^(q) saturates to 1.0, questions become "unlearnable" (no valid GRPO advantage for all-correct groups).
- Yet P(CC|CA)^(q) may still be imperfect (~0.7 median), indicating room for improvement that answer-only rewards cannot capture.
- Future research should explore mechanisms to accelerate P(CC|CA) improvement.

### 9.5 Distillation vs RLVR Trade-offs
- For models with poor initial (p_c, alpha, beta), distillation is more efficient than pure RLVR.
- Cold-start data / SFT alignment before RLVR can help rectify prior logic biases.
- SFT on RLVR-generated CoTs can nearly replicate RLVR performance at much lower cost -- practical implication for model deployment.

### 9.6 Domain-Specific Considerations
- **Code domains:** RLVR extends reasoning boundary even for distilled LLMs (execution feedback provides strong learning signal).
- **Math domains:** RLVR extends boundary for base LLMs (DAPO) but mainly improves Pass@1 for already-distilled models (Skywork-OR1-Math).
- **Cross-domain:** Training on one domain (math with integer answers) does not transfer to other domains (physics with free-form answers on Minerva).

### 9.7 Future Algorithm Design Principles
- New algorithms should more **directly incentivize correct reasoning paths**, not just correct answers.
- Potential directions: policy-gradient approaches, likelihood-based optimization, preference optimization frameworks.
- Process reward models (verifying intermediate steps) could complement outcome-based RLVR.
- Key principle: alleviate inherent logical biases in base LLMs.

---

## 10. Failure Mode Analysis (Case Studies)

The paper includes detailed case studies (Appendix A.8) showing how Qwen2.5-32B produces correct final answers through flawed reasoning:

### 10.1 AIME 2024 Examples
- **Rhombus on hyperbola:** Incorrectly assumes a rhombus is a square; gets correct answer 480 by coincidence.
- **Dodecagon rectangles:** Flawed counting method (incomplete diagonal classification, invalid rectangle assumption); gets correct answer 315 by coincidence.
- **Octagon coloring:** Invalid logical assumptions about adjacency conditions; gets correct answer 371 by coincidence.
- **Tetrahedron inscribed sphere:** Incorrect volume formula and wrong Cayley-Menger determinant entries; gets correct answer 104.

### 10.2 AIME 2025 Examples
- **Heptagon area:** Uses undefined quantities (EG) and invalid base calculations; gets correct answer 588.
- **Sawtooth + parabola intersection:** Generates Python code that solves wrong equation system and only considers one period; gets answer 259.

### 10.3 AMC23 Examples
- **Algebraic equation pairs:** Initial expansion error propagates; finds answer 1 through invalid assumptions.
- **Logarithm equation:** Incorrectly cancels terms and treats products as sums; gets answer 1.

### 10.4 Implications
- These case studies validate the necessity of CoT-Pass@K: base LLMs achieve surprisingly high Pass@K through fundamentally flawed reasoning.
- The LLM verifier (DeepSeek-R1-0528-Qwen3-8B) reliably identifies these critical errors.

---

## 11. Limitations Acknowledged

1. **LLM-as-verifier reliability:** Using an LLM to verify CoT correctness is imperfect; conflicting verification results exist across multiple queries.
2. **Theoretical scope:** Theorem 1 only explains the optimization process of RLVR but provides no guarantee for generalization; generalization is only observed empirically.
3. **DAPO reproduction gap:** Did not fully reproduce reported >50% Pass@1; achieved ~44% instead.
4. **Saturation problem:** After P(CA)^(q) saturates, remaining reasoning errors cannot be addressed purely through answer-correctness rewards.
5. **Manual verification cost:** Prohibitive to manually check all generated reasoning paths at scale.

---

## 12. Key Takeaways for Our Project

1. **RLVR genuinely improves reasoning quality**, not just sampling efficiency, when measured with appropriate metrics (CoT-Pass@K).
2. **The Logic Prior from pre-training is the foundation** -- RLVR amplifies existing correct-reasoning tendencies rather than creating them from scratch.
3. **Answer-only rewards have inherent limitations** -- they cannot fully resolve all reasoning errors, suggesting that process-level rewards or CoT verification could improve RLVR.
4. **Evaluation methodology matters enormously** -- the paper's core contribution is showing that the prior debate (RLVR helps vs. doesn't help) was partly an artifact of using answer-only metrics.
5. **SFT on RLVR-generated CoTs is a practical and effective strategy** for replicating RLVR benefits at lower cost.
6. **Challenging, contamination-free benchmarks are essential** for accurately measuring RLVR progress.
