# Literature Review: Collaborative RLVR for Robust Reasoning

## Research Area Overview

Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a dominant paradigm for training LLMs to perform mathematical reasoning. Pioneered by DeepSeek-R1 (2025), the approach uses rule-based correctness verification as rewards in RL training (typically GRPO), enabling reasoning capabilities to emerge without neural reward models. However, growing evidence shows that RLVR-trained models are brittle—they exploit dataset patterns, produce unfaithful reasoning chains, and fail under simple rephrasings or distribution shifts. Meanwhile, multi-agent debate has been shown to improve reasoning quality at inference time by having models critique and refine each other's solutions. This review covers three intersecting areas: (1) RLVR for mathematical reasoning, (2) multi-agent collaborative reasoning, and (3) robustness and faithfulness of LLM reasoning.

---

## Key Papers

### RLVR Core

#### DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- **Authors:** DeepSeek-AI (2025), arXiv:2501.12948
- **Key Contribution:** Demonstrates that RL alone (GRPO with rule-based rewards) can elicit strong reasoning from a base model (R1-Zero), and that a 4-stage pipeline (cold-start SFT → reasoning RL → rejection sampling SFT → all-scenario RL) achieves OpenAI-o1-level performance.
- **Methodology:** GRPO algorithm with group-relative advantage estimation (no critic model). Rewards: binary accuracy + format compliance. Base model: 671B MoE (37B activated).
- **Datasets:** AIME 2024, MATH-500, GPQA Diamond, LiveCodeBench, Codeforces, plus 800k SFT samples (600k reasoning + 200k general).
- **Results:** R1-Zero: 71.0% AIME pass@1 (from 15.6% base). R1: 79.8% AIME, 97.3% MATH-500. Distilled 7B model surpasses QwQ-32B.
- **Relevance:** Foundational paper establishing the RLVR paradigm. GRPO is the standard algorithm. Key design choices: rule-based rewards avoid reward hacking; PRM and MCTS were unsuccessful at scale.

#### RL with Verifiable Rewards Implicitly Incentivizes Correct Reasoning (Wen et al., 2025)
- **arXiv:** 2506.14245
- **Key Contribution:** Proves theoretically (Theorem 1) that GRPO implicitly incentivizes correct chain-of-thought under a "Logic Prior" assumption—correct CoTs are more likely to produce correct answers than incorrect CoTs.
- **Methodology:** Novel CoT-Pass@K metric that requires both correct answer AND correct reasoning. LLM-as-a-CoT-Judge with 3-way verification.
- **Results:** CoT-Pass@K shows persistent RLVR advantage over base models across all K (up to 1024) on AIME 2024/2025, while standard Pass@K misleadingly shows base models catching up.
- **Relevance:** Establishes that RLVR does improve reasoning quality (not just sampling efficiency), but with limitations—P(CC|CA) saturates at ~0.7 even after full training. Motivates process-level rewards.

#### Limits of Generalization in RLVR (Alam & Rastogi, 2025)
- **arXiv:** 2510.27044
- **Key Contribution:** Demonstrates rigorously that RLVR improvements often arise from reinforcing superficial heuristics rather than genuine reasoning, using two fully-verifiable combinatorial tasks.
- **Methodology:** Activity Scheduling (greedy) and LIS (dynamic programming) with unique verifiable solutions. GRPO on Qwen2.5-7B-Instruct. Random Forest regression shows RLVR outputs predictable from simple input features (R² up to 0.87).
- **Results:** Answer-only reward collapses CoT on LIS. Models emit decorative "sorting prefaces" with only ~2% accuracy. Hints provide zero benefit. Curriculum on sub-goals causes catastrophic collapse.
- **Relevance:** Core motivation for our research—demonstrates the brittleness problem. Recommends sequence-aware rewards, Acc_ids metrics, and benchmarks with unique verifiable solutions.

#### SimpleRL-Zoo (Zeng et al., 2025)
- **arXiv:** 2503.18892
- **Key Contribution:** Systematic study of zero RL (GRPO from base models) across 10 diverse models. Shows increased response length ≠ genuine reasoning emergence. First observation of verification behavior in non-Qwen small models.
- **Results:** All 10 models improve accuracy. Training data difficulty must match model capability. SFT cold-start limits post-RL exploration.
- **Relevance:** Reveals that reasoning quality varies dramatically across model families—motivates collaborative approaches where stronger models' patterns could benefit weaker ones.

#### Additional RLVR Papers
- **AceReason-Nemotron** (2505.16400): Advances math+code reasoning via RL, showing clear Pass@K improvements on LiveCodeBench.
- **Trust, But Verify** (2505.13445): Self-verification approach to RLVR.
- **Dual-Token Constraints for RLVR** (2507.15778): Stabilizes knowledge while promoting reasoning.
- **Uncertainty-Aware Advantage Shaping** (2510.10649): Unlocks exploration in RLVR via uncertainty estimation.
- **RL for Reasoning with One Training Example** (2504.20571): Shows a single example can rival large-scale RLVR, questioning whether RLVR induces genuinely new reasoning.
- **Med-RLVR** (2502.19655): Extends RLVR to medical reasoning from a 3B base model.

---

### Multi-Agent Collaborative Reasoning

#### Improving Factuality and Reasoning through Multiagent Debate (Du et al., 2023)
- **arXiv:** 2305.14325
- **Key Contribution:** Foundational multi-agent debate paper. N agents independently answer, then iteratively critique/refine each other's responses over R rounds. Requires only black-box access.
- **Results:** Arithmetic 67→82%, GSM8K 77→85% (3 agents, 2 rounds, GPT-3.5). Debate substantially outperforms reflection and majority voting. Gains from debate and CoT are additive. Crucially, debate can correct errors even when ALL agents are initially wrong.
- **Cost:** ~6x compute (3 agents × 2 rounds). Diminishing returns after 4 rounds.
- **Relevance:** Establishes that collaborative verification improves reasoning at inference time. Motivates training models to internalize debate benefits.

#### Self-Debate Reinforcement Learning (SDRL) (Liu et al., 2026)
- **arXiv:** 2601.22297
- **Key Contribution:** Most directly relevant paper. Trains a single LLM to be both a strong standalone solver AND effective debate participant via joint RL optimization on initial and debate-conditioned responses.
- **Methodology:** Uses model's own diverse rollouts as debate partners (self-debate). Frequency-based pairing identifies dominant competing beliefs. Built on DAPO/GRPO with verl framework. Training data: DAPO-Math-17K.
- **Theoretical Contribution:** Extends DCM/Bayesian framework to show that training "private critique advantage" breaks the martingale neutrality of standard debate, inducing positive drift toward correctness (with logarithmic diminishing returns).
- **Results:** Qwen3-4B-Base: debate accuracy from 52.9→57.7 (Δ from 0.1→3.5). Single-agent maj@32 also improves (56.2→60.2). Gains generalize across Sparse MAD, Centralized MAD, and varying agent counts.
- **Relevance:** Closest paper to our research. Demonstrates that collaborative RLVR is feasible and that debate training improves both collaborative and individual reasoning.

#### Multi-Agent Consensus Alignment (MACA) (Samanta et al., 2026)
- **arXiv:** 2509.15172
- **Key Contribution:** Post-trains LMs using debate-derived consensus signals. Compares MV-SFT, MV-GRPO, MV-DPO, and MV-KTO—preference learning (DPO/KTO) on debate traces is most effective.
- **Results:** Up to +26.87% MATH improvement and +42.73% MathQA. Strong generalization to unseen benchmarks (+16.3% GPQA). No ground-truth labels needed.
- **Relevance:** Demonstrates that multi-agent debate signals can be internalized via RL post-training, with DPO/KTO over debate traces as the most effective approach.

#### Debate Only When Necessary (DOWN) (Eo et al., 2025)
- **arXiv:** 2504.05047
- **Key Contribution:** Adaptive debate framework using confidence-based thresholding. High-confidence responses skip debate.
- **Results:** Comparable accuracy with up to 6x fewer agent calls. Lower correct-to-incorrect flip rate than always-on debate.
- **Relevance:** Demonstrates that selective collaboration outperforms always-on debate—informs when to trigger collaborative reasoning.

#### Additional Debate Papers
- **Debate or Vote** (2508.17536): Comparative analysis of debate vs. voting in multi-agent settings.
- **Inter-Consistency of LLM Collaboration** (2305.11595): In-depth analysis of debate dynamics.
- **Corex** (2310.00280): Multi-model collaboration pushing reasoning boundaries.
- **MARS** (2509.20502): Efficient multi-agent collaboration for LLM reasoning.
- **DynaDebate** (2601.05746): Dynamic path generation to break homogeneity in debate.
- **Stop Overvaluing Multi-Agent Debate** (2507): Argues for model heterogeneity and rethinking evaluation.

---

### Robustness and Faithfulness

#### GSM-Symbolic (Mirzadeh et al., 2024)
- **arXiv:** 2410.05229
- **Key Contribution:** Reveals fragility of LLM math reasoning via symbolic templates from GSM8K. Models are sensitive to irrelevant changes (names, numbers) and catastrophically fail on NoOp variants (irrelevant information).
- **Results:** GSM-NoOp causes up to 65% accuracy drops. Even o1-preview drops 17.5%. Performance variance across instantiations is significant for all models.
- **Relevance:** Provides both motivation (reasoning is brittle) and evaluation methodology (symbolic templates, NoOp testing) for our research. GSM-Symbolic data available on HuggingFace.

#### Can LMs Perform Robust Reasoning with Noisy Rationales? (2024)
- **arXiv:** 2410.23856
- **Key Contribution:** Studies robustness of reasoning under noisy/corrupted chain-of-thought.
- **Relevance:** Directly relevant to understanding when collaborative verification can help identify and correct reasoning errors.

#### Chain-of-Thought Reasoning Is Not Always Faithful
- Multiple papers address this (2305.18248, 2407.15647, 2505.10978, 2509.11082):
  - CoT can be decorative rather than reflecting the actual computation
  - RLVR models show ~0.7 P(CC|CA) even after convergence
  - Evaluating GRPO and DPO for faithful CoT shows preference learning may help
- **Relevance:** Collaborative reasoning (debate/discussion) forces models to externalize and defend their reasoning, potentially improving faithfulness.

---

## Common Methodologies

### RL Algorithms
- **GRPO (Group Relative Policy Optimization):** Standard for RLVR. No critic model, group-based advantage normalization. Used in DeepSeek-R1, SimpleRL-Zoo, SDRL.
- **DAPO (Decoupled Alignment via Policy Optimization):** GRPO variant with asymmetric clipping and dynamic sampling. Used in SDRL, RLVR correct reasoning paper.
- **DPO/KTO:** Preference optimization on debate traces (MACA). DPO/KTO outperform GRPO for internalizing debate signals.

### Multi-Agent Frameworks
- **Decentralized MAD:** Each agent sees all others' responses, generates independently.
- **Sparse MAD:** Agents see only a subset of peers.
- **Centralized MAD:** A central aggregator combines responses.
- **Self-Debate:** Model debates against its own diverse rollouts (SDRL).

### Reward Design
- **Rule-based accuracy rewards:** Binary correctness verification (standard).
- **Format rewards:** Enforce reasoning structure (think tags, etc.).
- **Sequence-aware rewards:** Verify intermediate reasoning steps (recommended by Alam & Rastogi).
- **Consensus rewards:** Agreement among debate participants as reward signal (MACA).

---

## Standard Baselines

| Baseline | Description | Typical Performance |
|----------|-------------|-------------------|
| Single-agent greedy | Standard inference | Baseline |
| Majority voting (maj@K) | K samples, majority answer | +5-15% over greedy |
| Self-reflection | Model critiques own answer | Mixed (+/- 5%) |
| Multi-agent debate | N agents, R rounds | +8-15% over single agent |
| GRPO/DAPO (standard RLVR) | RL with correctness rewards | +10-30% over base on math |
| SFT distillation | SFT on RLVR-generated data | Approaches RLVR performance |

---

## Evaluation Metrics

| Metric | When to Use | Notes |
|--------|------------|-------|
| Pass@1 | Standard accuracy | Can be misleading for small answer spaces |
| Pass@K | Reasoning coverage | Base models catch up; insufficient alone |
| CoT-Pass@K | True reasoning quality | Requires CoT verifier; recommended |
| Acc_ids | Sequence-level correctness | For tasks with verifiable intermediate steps |
| Self-consistency (SC) | Reliability of reasoning | Majority vote accuracy |
| Debate delta (Δ) | Debate improvement | Post-debate acc minus pre-debate acc |
| P(CC\|CA) | Reasoning faithfulness | Probability of correct CoT given correct answer |

---

## Datasets in the Literature

| Dataset | Used In | Task | Size |
|---------|---------|------|------|
| GSM8K | Nearly all papers | Grade school math | 7.5K train, 1.3K test |
| MATH | DeepSeek-R1, SimpleRL-Zoo, MACA | Competition math | 12.5K total |
| MATH-500 | SDRL, RLVR correct reasoning | Math evaluation subset | 500 test |
| DAPO-Math-17K | SDRL, RLVR correct reasoning | RLVR training | 17K train |
| AIME 2024/2025 | DeepSeek-R1, SDRL | Competition math evaluation | 30 per year |
| AMC 2023 | SDRL | Competition math evaluation | ~25 |
| GSM-Symbolic | GSM-Symbolic paper | Robustness evaluation | 5K per variant |
| LiveCodeBench | DeepSeek-R1, RLVR correct reasoning | Code reasoning | Evolving |

---

## Gaps and Opportunities

1. **No joint RLVR + debate training with robustness evaluation:** SDRL shows collaborative RLVR works, but doesn't evaluate on robustness benchmarks (GSM-Symbolic, NoOp). Our research fills this gap.

2. **Limited exploration of model heterogeneity in debate:** Most debate work uses homogeneous agents (same model). Collaborative RLVR could benefit from training models with complementary reasoning strategies.

3. **Process rewards in collaborative settings:** RLVR correct reasoning paper shows answer-only rewards have inherent ceilings (P(CC|CA) ~0.7). Debate-based consensus could serve as a process-level reward signal.

4. **Adaptive collaboration during training:** DOWN shows selective collaboration works at inference; this could be integrated into RLVR training (debate only on hard problems where the model disagrees with itself).

5. **Robustness-specific training objectives:** None of the RLVR papers explicitly optimize for robustness to rephrasing or irrelevant information. GSM-Symbolic variants could be used as training data.

6. **Faithfulness through externalization:** The faithfulness literature shows CoT is often decorative. Collaborative reasoning forces externalization and defense of reasoning—potentially improving faithfulness as a byproduct.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **Training:** DAPO-Math-17K (standard RLVR training set, used by SDRL and others)
2. **Evaluation (standard):** MATH-500, AIME 2024 (standard math benchmarks)
3. **Evaluation (robustness):** GSM-Symbolic (symbolic variants), GSM-NoOp (irrelevant info robustness)
4. **Evaluation (baseline):** GSM8K test set

### Recommended Baselines
1. Standard RLVR (GRPO/DAPO) without collaboration
2. RLVR + inference-time debate (no collaborative training)
3. Majority voting on RLVR model outputs
4. SDRL (self-debate RL) as the closest prior approach

### Recommended Metrics
1. **Pass@1 and maj@K** for standard accuracy
2. **CoT-Pass@K** for reasoning quality (if CoT verifier available)
3. **Debate delta (Δ)** for collaborative improvement
4. **GSM-Symbolic variance** for robustness to rephrasing
5. **GSM-NoOp accuracy** for robustness to irrelevant information
6. **P(CC|CA)** for reasoning faithfulness (if verifier available)

### Methodological Considerations
- **Model scale:** 3B-8B models are practical for research; Qwen2.5 and Llama families well-studied
- **Training framework:** verl (used by SDRL, SimpleRL-Zoo) is the standard
- **Debate pairing:** Frequency-based pairing (SDRL-freq) outperforms random
- **Joint optimization:** Train initial and debate responses in a single batch (SDRL approach)
- **Evaluate on distribution shifts:** Use disjoint difficulty levels and symbolic variants
- **Monitor training dynamics:** Track response length, entropy, P(CC|CA), and debate delta throughout training
