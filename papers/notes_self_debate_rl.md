# Notes: Prepare Reasoning Language Models for Multi-Agent Debate with Self-Debate Reinforcement Learning

**Paper:** arXiv 2601.22297v1 (January 29, 2026)
**Authors:** Chenxi Liu, Yanshuo Chen, Ruibo Chen, Tianyi Xiong, Tong Zheng, Heng Huang (University of Maryland, College Park)

---

## 1. Core Problem

Standard RLVR methods train LLMs to solve problems in isolation (single trajectories), without preparing them for collaborative environments where they must process and synthesize diverse rationales from other agents. This creates a mismatch between training (single-agent) and inference (multi-agent debate). Existing MAD methods often fail to outperform strong majority-voting baselines, and even state-of-the-art LLMs struggle to effectively incorporate conflicting opinions during debate.

Previous attempts to train debate behaviors either (a) optimize an entire multi-agent system with specialized roles, significantly increasing training costs and deployment complexity, or (b) train a separate aggregator model. Neither produces a single general-purpose LLM that is both a strong standalone solver and an effective debate participant.

---

## 2. How They Combine Multi-Agent Debate with RL Training

SDRL (Self-Debate Reinforcement Learning) is an **online RL framework** that trains a **single policy** to jointly improve:
1. **Single-agent reasoning** -- standard RLVR objective
2. **Private critique ability** -- the capacity to discriminate between different opinions during debate

The key insight is that in MAD, each agent's improvement comes from two sources:
- **Majority voting amplification** (statistical ensemble effect)
- **Private critique** (the agent's ability to revise its answer after reading peers' reasoning)

SDRL specifically targets improving private critique, which the theoretical analysis shows can break the martingale neutrality of standard debate and induce positive drift toward the correct answer.

### Training pipeline (per iteration):
1. For each prompt q, sample n initial responses from the current policy
2. Score them with verifiable rewards (+1 correct, -1 incorrect)
3. Construct a debate pair from these initial responses
4. Build a debate prompt (two-turn conversation) and sample second-round responses conditioned on the debate context
5. Score second-round responses with the same verifiable rewards
6. **Jointly optimize** both initial and debate-conditioned responses in a single training batch using the RL optimizer

This joint optimization is critical: it ensures the model improves as both a standalone solver and a debate participant simultaneously.

---

## 3. The Self-Debate Mechanism

### Debate Pair Construction

Given initial rollouts O(q) = {o_i} for a prompt q, SDRL constructs a debate pair P(o1, o2) using one of two pairing strategies:

1. **Random pairing (SDRL-rand):** Samples two rollouts uniformly from O(q). Produces a broad distribution of pair diversity -- some pairs genuinely disagree, others share the same answer but differ in reasoning trajectories (useful for learning confirmation vs. unconditional revision).

2. **Frequency-based pairing (SDRL-freq):** Identifies the most common answer a1 and second most common answer a2 among rollouts, then selects one response with each answer. This consistently exposes the model to the dominant competing beliefs of its current policy, yielding sharper disagreement and requiring stronger private critique ability. The order of a1 and a2 in the debate prompt is randomized to prevent positional heuristics.

### Debate Prompt Formulation

The debate prompt serializes two candidate responses into a two-turn conversation format:
- Turn 1 (User): Question -> Turn 1 (Assistant): First candidate response
- Turn 2 (User): Presents the second candidate response and instructs the model to:
  1. Identify key conflicts between the two solutions
  2. Critique conflicts step by step (incorrect steps, hidden assumptions, arithmetic errors)
  3. Acknowledge and switch if the other response is correct, otherwise defend
  4. Provide a clean, self-contained step-by-step resolution
  5. Output the final answer

### Why "Self-Debate"

The debate is between the model's own sampled responses (not between different models). The model plays both the initial solver and the debate participant, learning to critique its own diverse reasoning trajectories. This is computationally efficient -- only a single model is trained, and it serves all roles during MAD at inference time.

---

## 4. Training Methodology

### Base RL Algorithm

- Uses **GRPO** (Group Relative Policy Optimization) framework
- Specifically adopts **DAPO** (Decoupled Clip and Dynamic Sampling Policy Optimization) for stability with long chain-of-thought reasoning
- DAPO features: asymmetric clipping (epsilon_low=0.2, epsilon_high=0.28), dynamic sampling constraints, KL coefficient = 0
- Advantages computed via group normalization (Eq. 6)

### SDRL Algorithm (Algorithm 1)

```
For each training iteration:
  1. Sample mini-batch of prompts
  2. For each prompt:
     a. Sample n initial rollouts from current policy
     b. Compute rewards and advantages for initial rollouts
     c. Add initial rollouts to training batch
     d. Select debate pair from initial rollouts, build debate prompt
     e. Sample n_d second-round (debate) rollouts conditioned on debate prompt
     f. Compute rewards and advantages for debate rollouts
     g. Add debate rollouts to training batch
  3. Update policy using RL optimizer on combined batch
```

### Filtering and Efficiency

- Follow DAPO to oversample prompts and filter those whose responses all have zero advantage (ensuring diverse final answers per prompt)
- Debate responses with zero advantage are also filtered out for efficiency
- Most debate training occurs early in training; as the model improves, debating on the training set becomes easier and an increasing fraction receives zero advantage
- No oversampling applied for debate responses, keeping additional computational cost modest

### Hyperparameters

- Learning rate: 1e-6, linear warm-up over first 10 rollout steps
- Prompt batch size: 256, 8 responses per prompt for initial rollouts
- Debate responses: 4 per pair for Qwen2.5-3B, 8 per pair for Qwen3-4B-Base
- Max response length: 8,196 tokens
- Overlong buffer: 2,048 tokens with penalty factor 1
- Training mini-batch size: 128 (16 gradient updates per rollout step)
- 200 prompt generation steps (3,200 policy updates for DAPO baseline; SDRL has more due to debate training)
- At each step, 128 prompts randomly sampled for debate pair construction
- Sparse reward: +1 correct, -1 incorrect
- Framework: verl (HybridFlow)

---

## 5. Theoretical Analysis

### Bayesian Belief Update Model

Building on the Dirichlet Compound Multinomial (DCM) framework of Choi et al. (2025):

- Each agent i maintains Dirichlet parameters alpha_{i,t} at round t
- Standard Bayesian debate induces a **martingale** over each agent's belief in the correct answer (debate alone does not improve expected correctness)
- SDRL introduces **private critique** beta_{i,t} (a pseudo-count vector representing the agent's private analysis of the debate context)

### Critique-Augmented Belief Update (Definition 4.1)

alpha_{i,t} = alpha_{i,t-1} + beta_{i,t-1} + w_i * c_{i,t}

where c_{i,t} is the neighbor count vector and w_i controls social signal strength.

### Key Theorems

**Theorem 4.4 (Critique induces belief drift):** Under mean-consistency conditions, the expected belief in the correct answer increases by delta_{i,t-1} / (||alpha_{i,t-1}||_1 + C_i), where delta is the "critique advantage" -- the extent to which private critique allocates more belief to the correct answer than the prior.

**Lemma 4.5 (Diminishing returns):** Even under sustained positive critique advantage, improvement accumulates only logarithmically in the number of rounds.

**Lemma 4.7 (Correlation shrinks effective ensemble size):** As debate rounds progress, agents condition on increasingly similar contexts, raising answer correlation and effectively reducing ensemble size N_eff = N / (1 + (N-1)*rho).

### Explanation of Rise-Then-Fall Pattern

The theory explains the common empirical observation that debate performance:
- **Improves in early rounds** (positive critique drift + low inter-agent correlation)
- **Peaks quickly and can decline later** (marginal drift shrinks logarithmically + correlation erodes majority-vote amplification)

### Role of Majority Voting vs. Private Critique

When delta_{i,t} = 0 (no critique training signal), the framework reduces to neutral martingale belief evolution -- accuracy is governed purely by vote amplification. SDRL helps by increasing delta, which induces positive drift. However, even sustained advantage yields diminishing per-round gains.

---

## 6. Datasets and Benchmarks

### Training Data
- **DAPO-Math-17K** dataset (Yu et al., 2025) -- mathematical reasoning problems

### Evaluation Benchmarks
- **MATH500** (Hendrycks et al., 2021) -- mathematical reasoning
- **AMC 2023** -- American Mathematics Competition
- **AIME 2024** -- American Invitational Mathematics Examination
- **AIME 2025** -- American Invitational Mathematics Examination

### Base Models
- **Qwen2.5-3B** (smaller model)
- **Qwen3-4B-Base** (stronger model)

### Evaluation Setup
- **Multi-Agent Debate:** Primarily decentralized MAD with N=5 agents, T=1 debate round; temperature=1.0, top-p=0.9; average over 5 independent runs
- **Single-Agent:** mean@K (average accuracy) and maj@K (majority-vote accuracy); K=32 for AMC/AIME, K=4 for MATH500
- Additional MAD frameworks tested: Sparse MAD, Centralized MAD

---

## 7. Key Results on Reasoning Quality

### Multi-Agent Debate Results (Table 1, N=5, T=1)

**Qwen2.5-3B:**
| Method | Avg Maj | Avg Debate | Avg Delta |
|--------|---------|------------|-----------|
| DAPO | 32.8 | 33.6 | 0.8 |
| +SDRL-rand | 33.6 | 35.8 | 2.2 |
| +SDRL-freq | 34.1 | 35.8 | 1.7 |

**Qwen3-4B-Base:**
| Method | Avg Maj | Avg Debate | Avg Delta |
|--------|---------|------------|-----------|
| DAPO | 52.8 | 52.9 | 0.1 |
| +SDRL-rand | 53.7 | 55.6 | 1.9 |
| +SDRL-freq | 54.2 | 57.7 | 3.5 |

Key findings:
- SDRL consistently improves both post-debate accuracy and first-round majority vote
- Gains most pronounced on harder benchmarks (AIME24: debate accuracy jumps from 28.3 to 36.0 with SDRL-freq on Qwen3-4B-Base; delta from 1.6 to 7.3)
- DAPO baseline shows mixed debate behavior (degradation on MATH500 and AIME25 for Qwen3-4B-Base)
- SDRL-freq generally yields stronger and more robust improvements than SDRL-rand

### Single-Agent Results (Table 2)

SDRL also improves standalone reasoning (no debate needed):
- Qwen2.5-3B: avg mean@32 from 30.8 to 32.3, avg maj@32 from 36.2 to 37.5
- Qwen3-4B-Base: avg mean@32 from 51.1 to 52.4, avg maj@32 from 56.2 to 60.2

This is notable: training for debate simultaneously strengthens individual problem-solving ability.

---

## 8. How Debate Improves Reasoning Robustness

### Multiple Debate Rounds (Figure 1)
- SDRL achieves higher accuracy across ALL debate rounds compared to DAPO
- SDRL yields larger improvements as number of rounds increases
- Performance increases in early rounds then declines (consistent with theory)
- Exception: MATH500 where performance can degrade due to correct answers being short while incorrect ones are long, biasing the debate context

### Varying Number of Agents (Table 3)
- SDRL consistently outperforms DAPO across N=3, 5, 7 agents
- Most dramatic: Qwen3-4B-Base N=3, SDRL achieves avg debate gain delta=4.7 vs DAPO delta=0.9
- 7-agent setting shows some degradation due to context window limitations requiring truncation

### Robustness Across Debate Frameworks (Table 4)
- **Sparse MAD:** SDRL improves avg debate accuracy (Qwen3-4B-Base: 53.5 -> 58.6) and gain (2.6 -> 5.2)
- **Centralized MAD:** SDRL improves avg debate accuracy (Qwen3-4B-Base: 50.2 -> 54.9) and gain (2.9 -> 5.5)
- SDRL's improvements generalize beyond the decentralized MAD setting it was trained in

### Why Debate Helps After SDRL Training

After SDRL training, debate more reliably yields positive improvements over first-round voting because:
1. The model has learned to perform effective **private critique** -- identifying errors in its own and others' reasoning
2. The model knows **when to revise** (switch to correct answer) vs. **when to defend** (maintain correct answer against incorrect alternatives)
3. The system benefits most on **difficult problems** where correcting erroneous trajectories via critique is essential

---

## 9. Comparison to Standard RLVR Approaches

| Aspect | Standard RLVR (DAPO) | SDRL |
|--------|---------------------|------|
| Training paradigm | Single-agent trajectories only | Joint single-agent + debate-conditioned trajectories |
| Debate preparation | None (models solve in isolation) | Explicit training with debate pairs and critique |
| MAD delta (Qwen3-4B avg) | 0.1 | 3.5 (SDRL-freq) |
| Single-agent performance | Baseline | Improved (side benefit of debate training) |
| Number of models trained | 1 | 1 (same single model) |
| Deployment complexity | Low | Low (same single model used for all agents) |
| Additional training cost | Baseline | Modest (most debate training occurs early; filtered) |

Key differentiators from other approaches:
- Unlike multi-agent RL systems (MARTI, MARFT, SPIRAL), SDRL trains a **single model** that serves all debate roles
- Unlike aggregator-based approaches, SDRL does not require a separate aggregation model
- Unlike standard RLVR, SDRL explicitly prepares models for multi-agent interaction at inference time
- The debate training signal comes from the model's own diverse rollouts (self-debate), not from external debate partners

### Reduced Budget Experiment (Table 5)
- SDRL-freq with only 125 training steps outperforms fully-trained DAPO (200 steps) in debate performance
- This demonstrates gains come from training on debate data, not simply from more compute

---

## 10. Failure Modes and Limitations

### MATH500 Degradation Over Rounds
- On easy benchmarks (MATH500 for Qwen3-4B), correct answers are often retrieved from memorized knowledge and are SHORT
- Incorrect reasoning traces tend to be LONG (avg 7121 tokens incorrect vs 2227 correct at round 0)
- Long incorrect responses dominate the shared debate context, biasing subsequent rounds toward incorrect trajectories
- This causes accuracy to DROP over multiple debate rounds on easy problems

### Context Window Constraints
- With N=7 agents, max response tokens must be reduced to fit the debate sequence within the model context window
- This causes truncated/incomplete responses and can degrade performance

### Diminishing Returns
- Theoretical and empirical: debate improvement accumulates only logarithmically in the number of rounds
- Increasing inter-agent correlation over rounds reduces effective ensemble size

---

## 11. Code and Data Availability

- **Training framework:** verl (HybridFlow) -- arXiv:2409.19256 (Sheng et al., 2024)
- **Training dataset:** DAPO-Math-17K (Yu et al., 2025)
- **MAD evaluation codebase:** From Choi et al. (2025)
- **Single-agent evaluation:** VERL framework with MATH-VERIFY for answer extraction
- **No explicit code release URL mentioned** in the paper (preprint status as of January 2026)

---

## 12. Relevance to Collaborative RLVR for Robust Reasoning

This paper is highly relevant to our project because:

1. **Bridges RLVR and multi-agent collaboration:** SDRL is a concrete method for training models that benefit from collaborative reasoning, directly addressing the gap between isolated RLVR training and multi-agent inference.

2. **Theoretical foundation for debate benefits:** The DCM-based analysis provides formal conditions under which debate helps (positive critique advantage), when it saturates (logarithmic diminishing returns), and when it hurts (correlation shrinking effective ensemble size).

3. **Efficient single-model approach:** Unlike complex multi-agent RL systems, SDRL trains one model that serves all roles, making it practical and deployable.

4. **Joint optimization preserves standalone ability:** The critical finding that debate training also improves single-agent performance suggests no trade-off between collaborative and independent reasoning.

5. **Private critique as the key mechanism:** The theoretical identification of "private critique" as the driver of debate gains (rather than mere majority voting) provides a concrete training target for improving collaborative reasoning.

6. **Self-play flavor:** Using the model's own diverse rollouts as debate partners is a form of self-play that does not require external models or human feedback for the debate component.

7. **Practical design choices:** Frequency-based pairing, randomizing order to prevent positional bias, filtering zero-advantage debate responses -- all are implementable techniques for our work.
