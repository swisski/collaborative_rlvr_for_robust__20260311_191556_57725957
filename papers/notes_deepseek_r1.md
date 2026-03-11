# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

**Paper:** arXiv:2501.12948v1 [cs.CL], 22 Jan 2025
**Authors:** DeepSeek-AI (research@deepseek.com)
**Pages:** 22

---

## 1. High-Level Summary

DeepSeek-R1 is a reasoning-focused LLM trained via large-scale reinforcement learning (RL). The paper introduces two models:

- **DeepSeek-R1-Zero**: Trained with pure RL (GRPO) directly on the base model (DeepSeek-V3-Base) -- no SFT whatsoever. Demonstrates that reasoning capabilities can emerge from RL alone. Suffers from poor readability and language mixing.
- **DeepSeek-R1**: Uses a multi-stage pipeline with cold-start data + iterative RL + SFT + second RL stage. Achieves performance comparable to OpenAI-o1-1217 on reasoning tasks.

Additionally, 6 distilled dense models (1.5B, 7B, 8B, 14B, 32B, 70B) based on Qwen2.5 and Llama3 are released.

---

## 2. Base Model

- **DeepSeek-V3-Base** is used as the starting point for both R1-Zero and R1.
- Architecture: Mixture-of-Experts (MoE)
- Total parameters: **671B**
- Activated parameters: **37B**

---

## 3. GRPO Algorithm (Group Relative Policy Optimization)

GRPO (from Shao et al., 2024 -- DeepSeekMath paper) is the core RL algorithm. Key design choices:

- **No critic model**: Unlike standard PPO, GRPO foregoes the critic (value) model, which would typically be the same size as the policy. This saves significant training cost.
- **Group-based advantage estimation**: For each question q, sample a group of G outputs {o1, o2, ..., oG} from the old policy. Compute advantages by normalizing rewards within the group:
  ```
  A_i = (r_i - mean({r_1, ..., r_G})) / std({r_1, ..., r_G})
  ```
- **Objective**: Clipped surrogate objective (PPO-style) with KL penalty against a reference policy:
  ```
  J_GRPO(theta) = E[1/G * sum_i min(ratio * A_i, clip(ratio, 1-eps, 1+eps) * A_i) - beta * D_KL(pi_theta || pi_ref)]
  ```
  where ratio = pi_theta(o_i|q) / pi_theta_old(o_i|q)
- **KL divergence**: Uses a specific form: D_KL = pi_ref/pi_theta - log(pi_ref/pi_theta) - 1
- Hyperparameters: epsilon (clipping), beta (KL penalty weight)

---

## 4. Reward Model Design

### DeepSeek-R1-Zero: Rule-Based Rewards Only

Two types of rewards (NO neural reward model):

1. **Accuracy rewards**: Rule-based verification of correctness.
   - Math: Requires final answer in specified format (e.g., boxed), then deterministic check.
   - Code (LeetCode): Compiler-based feedback using predefined test cases.

2. **Format rewards**: Enforces `<think>...</think>` tag structure for reasoning.

**Critical design decision**: No outcome-based or process-based neural reward models. Rationale:
- Neural reward models suffer from **reward hacking** at large-scale RL.
- Retraining reward models adds complexity and computational overhead.

### DeepSeek-R1 (Stage 4 -- RL for All Scenarios)

- For **reasoning data**: Same rule-based rewards (math, code, logic).
- For **general data**: Neural reward models to capture human preferences in complex scenarios.
  - **Helpfulness**: Evaluates only the final summary (not the reasoning chain) to avoid interfering with the reasoning process.
  - **Harmlessness**: Evaluates the entire response (reasoning + summary) to identify risks, biases, and harmful content.

### Language Consistency Reward (R1 Stage 2)

- Calculated as proportion of target-language words in the CoT.
- Ablations show slight performance degradation but significantly improved readability.
- Combined with accuracy reward by direct summation.

---

## 5. Training Pipeline

### DeepSeek-R1-Zero Pipeline
Single stage: RL (GRPO) directly on DeepSeek-V3-Base.

### DeepSeek-R1 Pipeline (4 Stages)

**Stage 1: Cold Start (SFT)**
- Collect thousands of long Chain-of-Thought (CoT) examples.
- Data collection methods:
  - Few-shot prompting with long CoT examples
  - Directly prompting models to generate detailed answers with reflection/verification
  - Gathering DeepSeek-R1-Zero outputs in readable format
  - Post-processing/refinement by human annotators
- Fine-tune DeepSeek-V3-Base on this data.
- Output format: `|special_token|<reasoning_process>|special_token|<summary>`
- Benefits over R1-Zero: Better readability, better potential performance from human priors.

**Stage 2: Reasoning-Oriented RL**
- Same large-scale RL process as R1-Zero (GRPO).
- Focus: Reasoning-intensive tasks (coding, math, science, logic) with well-defined solutions.
- Rewards: Accuracy (rule-based) + language consistency reward.
- Train until convergence on reasoning tasks.

**Stage 3: Rejection Sampling + SFT**
- Use converged RL checkpoint to generate new SFT data via rejection sampling.
- **Reasoning data (~600k samples)**:
  - Curate reasoning prompts, generate trajectories from RL checkpoint.
  - Expand beyond rule-based verifiable data: use generative reward model (DeepSeek-V3 judges ground-truth vs. model predictions).
  - Filter out CoT with mixed languages, long paragraphs, code blocks.
  - Sample multiple responses per prompt, retain only correct ones.
- **Non-reasoning data (~200k samples)**:
  - Writing, factual QA, self-cognition, translation.
  - Reuse portions of DeepSeek-V3 SFT dataset.
  - For some tasks, use DeepSeek-V3 to generate a potential CoT before answering.
  - Simple queries (e.g., "hello") get no CoT.
- **Total: ~800k samples**
- Fine-tune DeepSeek-V3-Base for **2 epochs** on this combined dataset.

**Stage 4: RL for All Scenarios**
- Second RL stage for alignment (helpfulness + harmlessness) AND continued reasoning improvement.
- Mixed prompt distribution: reasoning + general tasks.
- Reasoning: Rule-based rewards.
- General: Neural reward models (built on DeepSeek-V3 pipeline).
- Produces the final DeepSeek-R1 checkpoint.

---

## 6. Training Template

For DeepSeek-R1-Zero, a minimal template is used:

```
A conversation between User and Assistant. The user asks a question, and the Assistant
solves it. The assistant first thinks about the reasoning process in the mind and then
provides the user with the answer. The reasoning process and answer are enclosed within
<think> </think> and <answer> </answer> tags, respectively, i.e.,
<think> reasoning process here </think>
<answer> answer here </answer>.
User: {prompt}. Assistant:
```

Key: No content-specific biases (no mandating reflection, no promoting particular strategies). Only structural format constraints, to observe natural RL progression.

---

## 7. Distillation

- Fine-tune open-source models on the **800k samples** curated in Stage 3.
- **SFT only** -- no RL stage applied to distilled models (though RL would further improve them).
- Base models used:
  - Qwen2.5-Math-1.5B
  - Qwen2.5-Math-7B
  - Qwen2.5-14B
  - Qwen2.5-32B
  - Llama-3.1-8B
  - Llama-3.3-70B-Instruct (chosen over 3.1 for slightly better reasoning)

### Key Finding: Distillation vs. RL on Small Models

| Model | AIME 2024 (pass@1) | MATH-500 | GPQA Diamond | LiveCodeBench |
|-------|---------------------|----------|--------------|---------------|
| QwQ-32B-Preview | 50.0 | 90.6 | 54.5 | 41.9 |
| DeepSeek-R1-Zero-Qwen-32B (RL) | 47.0 | 91.6 | 55.0 | 40.2 |
| DeepSeek-R1-Distill-Qwen-32B (distill) | **72.6** | **94.3** | **62.1** | **57.2** |

**Conclusions:**
1. Distilling from a more powerful model yields far better results than RL on smaller models.
2. Smaller models doing large-scale RL require enormous compute and still may not match distillation.
3. However, pushing beyond intelligence boundaries still requires more powerful base models + larger-scale RL.

---

## 8. Evaluation Benchmarks and Metrics

### Benchmarks Used
- **Knowledge**: MMLU, MMLU-Redux, MMLU-Pro, C-Eval, CMMLU
- **Instruction Following**: IFEval
- **Long-context QA**: FRAMES
- **Science**: GPQA Diamond
- **Factual QA**: SimpleQA, C-SimpleQA
- **Software Engineering**: SWE-Bench Verified, Aider
- **Code**: LiveCodeBench (2024-08 to 2025-01), Codeforces (10 Div.2 contests), HumanEval-Mul
- **Math**: AIME 2024, MATH-500, CNMO 2024
- **Open-ended**: AlpacaEval 2.0 (LC-winrate), Arena-Hard

### Evaluation Setup
- Max generation length: **32,768 tokens**
- Default: **pass@k evaluation** with non-zero temperature (greedy decoding causes high repetition rates).
- Temperature: **0.6**, top-p: **0.95**
- k responses per question (typically 4-64 depending on test set size).
- pass@1 = (1/k) * sum(p_i) where p_i is correctness of i-th response.
- For AIME 2024: Also report **cons@64** (majority vote with 64 samples).
- Open-ended tasks: GPT-4-Turbo-1106 as judge (AlpacaEval 2.0, Arena-Hard). Only the final summary is fed to evaluation to avoid length bias.

### Baselines
- DeepSeek-V3
- Claude-Sonnet-3.5-1022
- GPT-4o-0513
- OpenAI-o1-mini
- OpenAI-o1-1217
- QwQ-32B-Preview (for distilled model comparison)

---

## 9. Key Results

### DeepSeek-R1-Zero Performance

| Benchmark | DeepSeek-R1-Zero | OpenAI-o1-mini | OpenAI-o1-0912 |
|-----------|------------------|----------------|----------------|
| AIME 2024 (pass@1) | 71.0 | 63.6 | 74.4 |
| AIME 2024 (cons@64) | 86.7 | 80.0 | 83.3 |
| MATH-500 | 95.9 | 90.0 | 94.8 |
| GPQA Diamond | 73.3 | 60.0 | 77.3 |
| LiveCodeBench | 50.0 | 53.8 | 63.4 |
| Codeforces (rating) | 1444 | 1820 | 1843 |

- AIME pass@1 improves from **15.6% to 71.0%** during RL training.
- With majority voting: **86.7%**, exceeding o1-0912.

### DeepSeek-R1 Performance (Full Model)

| Benchmark | DeepSeek-V3 | o1-mini | o1-1217 | DeepSeek-R1 |
|-----------|-------------|---------|---------|-------------|
| MMLU | 88.5 | 85.2 | 91.8 | **90.8** |
| MMLU-Pro | 75.9 | 80.3 | - | **84.0** |
| GPQA Diamond | 59.1 | 60.0 | 75.7 | **71.5** |
| DROP (3-shot F1) | 91.6 | 83.9 | 90.2 | **92.2** |
| IF-Eval | 86.1 | 84.8 | - | 83.3 |
| SimpleQA | 24.9 | 7.0 | 47.0 | **30.1** |
| FRAMES | 73.3 | 76.9 | - | **82.5** |
| AlpacaEval 2.0 (LC) | 70.0 | 57.8 | - | **87.6** |
| ArenaHard | 85.5 | 92.0 | - | **92.3** |
| LiveCodeBench | 36.2 | 53.8 | 63.4 | **65.9** |
| Codeforces (percentile) | 58.7 | 93.4 | 96.6 | **96.3** |
| Codeforces (rating) | 1134 | 1820 | 2061 | **2029** |
| SWE Verified | 42.0 | 41.6 | 48.9 | **49.2** |
| AIME 2024 (pass@1) | 39.2 | 63.6 | 79.2 | **79.8** |
| MATH-500 | 90.2 | 90.0 | 96.4 | **97.3** |
| CNMO 2024 | 43.2 | 67.6 | - | **78.8** |
| C-Eval | 86.5 | 68.9 | - | **91.8** |

### Distilled Model Results

| Model | AIME (pass@1) | AIME (cons@64) | MATH-500 | GPQA Diamond | LiveCodeBench | Codeforces Rating |
|-------|---------------|----------------|----------|--------------|---------------|-------------------|
| DeepSeek-R1-Distill-Qwen-1.5B | 28.9 | 52.7 | 83.9 | 33.8 | 16.9 | 954 |
| DeepSeek-R1-Distill-Qwen-7B | 55.5 | 83.3 | 92.8 | 49.1 | 37.6 | 1189 |
| DeepSeek-R1-Distill-Qwen-14B | 69.7 | 80.0 | 93.9 | 59.1 | 53.1 | 1481 |
| DeepSeek-R1-Distill-Qwen-32B | 72.6 | 83.3 | 94.3 | 62.1 | 57.2 | 1691 |
| DeepSeek-R1-Distill-Llama-8B | 50.4 | 80.0 | 89.1 | 49.0 | 39.6 | 1205 |
| DeepSeek-R1-Distill-Llama-70B | 70.0 | 86.7 | 94.5 | 65.2 | 57.5 | 1633 |

Notable: The 7B distilled model (55.5% AIME) surpasses QwQ-32B-Preview (50.0%). The 1.5B model (28.9% AIME, 83.9% MATH) outperforms GPT-4o and Claude-3.5-Sonnet on math benchmarks.

---

## 10. Emergent Reasoning Behaviors

### Self-Evolution Process (R1-Zero)
- Thinking time (response length) increases consistently throughout RL training.
- Not from external adjustments -- intrinsic development.
- Model generates hundreds to thousands of reasoning tokens.

### Emergent Behaviors
- **Reflection**: Model revisits and reevaluates previous steps.
- **Alternative exploration**: Model spontaneously explores different approaches to problem-solving.
- **Self-verification**: Model checks its own answers.
- These behaviors emerge naturally from RL, not from explicit programming.

### "Aha Moment"
- Observed in intermediate training versions of R1-Zero.
- The model learns to reallocate thinking time by reevaluating its initial approach.
- Example: Model says "Wait, wait. Wait. That's an aha moment I can flag here. Let's reevaluate this step-by-step..."
- Uses anthropomorphic tone to indicate rethinking.

### Drawbacks of R1-Zero
- Poor readability
- Language mixing (mixing multiple languages in output)
- Not suitable for direct user consumption without further refinement

---

## 11. Reasoning Quality / Faithfulness

- The paper does not deeply analyze faithfulness of the chain-of-thought, but notes:
  - R1-Zero's CoT can be chaotic, with mixed languages and code blocks.
  - Cold-start data in R1 enforces readable patterns with summary at end.
  - Stage 3 filtering removes CoT with mixed languages, long paragraphs, code blocks.
  - For helpfulness evaluation in Stage 4 RL, only the final summary is assessed to avoid interfering with the reasoning process.
  - For harmlessness, the entire response including reasoning is evaluated.
  - Few-shot prompting consistently degrades R1's performance (sensitive to prompts).
  - Zero-shot is recommended for optimal results.

---

## 12. Unsuccessful Attempts

### Process Reward Models (PRM)
Three limitations identified:
1. Hard to define fine-grained steps in general reasoning.
2. Determining correctness of intermediate steps is challenging. Automated annotation is unsatisfactory; manual annotation doesn't scale.
3. Model-based PRM inevitably leads to reward hacking; retraining adds complexity.
- PRM is good for reranking top-N responses or guided search, but advantages are limited vs. computational overhead during large-scale RL.

### Monte Carlo Tree Search (MCTS)
- Inspired by AlphaGo/AlphaZero.
- Approach: Break answers into parts, explore solution space, use pre-trained value model.
- Challenges:
  1. Token generation has exponentially larger search space than chess. Setting max extension limits leads to local optima.
  2. Value model quality directly impacts generation quality. Training a fine-grained value model for text is inherently difficult.
- Conclusion: MCTS can improve inference with a pre-trained value model, but iteratively boosting model performance through self-search remains a significant challenge.

---

## 13. Code and Model Availability

**Open-sourced models:**
- DeepSeek-R1-Zero
- DeepSeek-R1
- DeepSeek-R1-Distill-Qwen-1.5B
- DeepSeek-R1-Distill-Qwen-7B
- DeepSeek-R1-Distill-Qwen-14B
- DeepSeek-R1-Distill-Qwen-32B
- DeepSeek-R1-Distill-Llama-8B
- DeepSeek-R1-Distill-Llama-70B

API also available.

---

## 14. Limitations and Future Work

1. **General Capability**: R1 falls short of DeepSeek-V3 on function calling, multi-turn, complex role-playing, JSON output.
2. **Language Mixing**: Optimized for Chinese and English only. Other languages may trigger English reasoning regardless of query language.
3. **Prompt Sensitivity**: Few-shot prompting degrades performance. Zero-shot recommended.
4. **Software Engineering**: Long evaluation times limit RL data. No huge improvement over V3 on SWE benchmarks. Future plans: rejection sampling on SWE data, async evaluations during RL.
5. **Safety RL side effects**: R1 performs worse than V3 on Chinese SimpleQA due to tendency to refuse queries after safety RL. Without safety RL, accuracy would exceed 70%.

---

## 15. Key Takeaways for Our Research

1. **Pure RL can elicit reasoning**: R1-Zero proves that GRPO on a base model, with only rule-based rewards, can produce strong reasoning capabilities. No SFT or neural reward models needed for the core reasoning emergence.

2. **Rule-based rewards are critical**: Accuracy rewards (deterministic verification) + format rewards. Neural reward models are explicitly avoided for reasoning due to reward hacking concerns.

3. **Cold start data helps**: A small amount of high-quality long-CoT data before RL accelerates convergence and improves readability. "Thousands" of examples, not hundreds of thousands.

4. **Multi-stage pipeline is key**: Cold-start SFT -> Reasoning RL -> Rejection sampling + SFT (800k) -> All-scenario RL. Each stage serves a distinct purpose.

5. **Distillation > RL for small models**: The 800k samples from R1 distilled into Qwen-32B massively outperform doing RL directly on Qwen-32B-Base. But pushing the frontier requires large-scale RL on large base models.

6. **GRPO is the algorithm of choice**: Eliminates the value/critic model, making large-scale RL feasible. Advantage computed via group normalization of rewards.

7. **PRM and MCTS were unsuccessful**: Despite being theoretically appealing, both approaches had practical limitations at scale.

8. **800k SFT dataset composition**: ~600k reasoning (rejection sampled from RL checkpoint) + ~200k non-reasoning (from V3 pipeline). This is the dataset used for both Stage 3 SFT and distillation.
