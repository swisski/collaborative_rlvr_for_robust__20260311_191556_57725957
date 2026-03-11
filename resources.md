# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Collaborative RLVR for Robust Reasoning," including 33 papers, 5 datasets, and 6 code repositories.

---

## Papers
Total papers downloaded: **33**

### RLVR Core (13 papers)

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| DeepSeek-R1 | DeepSeek-AI | 2025 | papers/2501.12948v1_deepseek_r1.pdf | Foundational RLVR paper; GRPO; 4-stage pipeline |
| Reasoning Gym | - | 2025 | papers/2505.24760v2_reasoning_gym.pdf | RL environments for verifiable rewards |
| RLVR Implicitly Incentivizes Correct Reasoning | Wen et al. | 2025 | papers/2506.14245v2_rlvr_correct_reasoning.pdf | CoT-Pass@K metric; Logic Prior theorem |
| RLVR with Noisy Rewards | - | 2025 | papers/2510.00915v3_rlvr_noisy_rewards.pdf | Imperfect verifiers |
| Trust But Verify | - | 2025 | papers/2505.13445v1_trust_but_verify_rlvr.pdf | Self-verification in RLVR |
| Limits of Generalization in RLVR | Alam & Rastogi | 2025 | papers/2510.27044v2_limits_generalization_rlvr.pdf | Brittleness; heuristic exploitation |
| SimpleRL-Zoo | Zeng et al. | 2025 | papers/2503.18892v3_simplerl_zoo.pdf | Zero RL across 10 models |
| AceReason-Nemotron | NVIDIA | 2025 | papers/2505.16400v3_acereason_nemotron.pdf | Math+code reasoning via RL |
| Dual-Token RLVR | - | 2025 | papers/2507.15778v1_dual_token_rlvr.pdf | Stabilizing knowledge in RLVR |
| Uncertainty RLVR | - | 2025 | papers/2510.10649v1_uncertainty_rlvr.pdf | Uncertainty-aware advantage shaping |
| Med-RLVR | - | 2025 | papers/2502.19655v1_med_rlvr.pdf | Medical reasoning from 3B model |
| RL One Example | Wang et al. | 2025 | papers/2504.20571v3_rl_one_example.pdf | Single training example rivals RLVR |
| Adaptive Math Reasoning | - | 2025 | papers/2510.04617v2_adaptive_math_reasoning.pdf | Making math reasoning adaptive |

### Multi-Agent Debate (8 papers)

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Self-Debate RL (SDRL) | Liu et al. | 2026 | papers/2601.22297v1_self_debate_rl.pdf | **Most relevant**: joint RLVR + debate training |
| Improving Factuality via Debate | Du et al. | 2023 | papers/2305.14325v1_improving_factuality_debate_du.pdf | Foundational debate paper |
| Debate Only When Necessary | Eo et al. | 2025 | papers/2504.05047v2_debate_only_when_necessary.pdf | Adaptive/selective debate |
| Debate or Vote | - | 2025 | papers/2508.17536v2_debate_or_vote.pdf | Comparative analysis |
| Inter-Consistency via Debate | - | 2023 | papers/2305.11595v3_inter_consistency_debate.pdf | Debate dynamics analysis |
| Corex | - | 2023 | papers/2310.00280v3_corex_multi_model.pdf | Multi-model collaboration |
| MARS | - | 2025 | papers/2509.20502v1_mars_multi_agent.pdf | Efficient multi-agent collaboration |
| Self-Improvement via Debate (MACA) | Samanta et al. | 2026 | papers/2509.15172v3_self_improvement_debate.pdf | DPO/KTO on debate traces |
| DynaDebate | - | 2026 | papers/2601.05746v1_dynadebate.pdf | Breaking homogeneity in debate |

### Robustness (3 papers)

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| GSM-Symbolic | Mirzadeh et al. (Apple) | 2024 | papers/2410.05229v1_gsm_symbolic.pdf | Symbolic templates; NoOp; up to 65% drops |
| Robust Reasoning with Noisy Rationales | - | 2024 | papers/2410.23856v1_robust_reasoning_noisy.pdf | Robustness under noise |
| R3 Prompting | - | 2023 | papers/2310.16535v1_r3_prompting.pdf | Review, Rephrase, Resolve |

### Faithful Reasoning (6 papers)

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Measuring Faithfulness in CoT | - | 2023 | papers/2305.18248v2_measuring_faithfulness_cot.pdf | Reference hallucination detection |
| Dissociation of Faithful/Unfaithful | - | 2024 | papers/2405.15092v2_dissociation_faithful_unfaithful.pdf | Faithful vs unfaithful reasoning |
| Hardness of Faithful CoT | - | 2024 | papers/2407.15647v2_hardness_faithful_cot.pdf | Difficulty of achieving faithfulness |
| CoT Not Always Faithful | - | 2025 | papers/2505.10978v1_cot_not_always_faithful.pdf | GiGPO agent training |
| GRPO/DPO for Faithful CoT | - | 2025 | papers/2509.11082v1_grpo_dpo_faithful_cot.pdf | Comparing RL methods for faithfulness |
| DeepSeekMath / R1-Zero | - | 2024 | papers/2411.16532v1_deepseek_math_r1_zero.pdf | Continual RL (different than expected) |

### GRPO/RL Techniques (2 papers)

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Stable RL for Reasoning | - | 2025 | papers/2505.18086v1_stable_rl_reasoning.pdf | Stable RL training |
| S-GRPO | - | 2025 | papers/2505.07686v2_s_grpo_early_exit.pdf | Early exit via RL |

### Deep-Read Notes Available
- papers/notes_deepseek_r1.md (22 pages, comprehensive)
- papers/notes_self_debate_rl.md (22 pages, comprehensive)
- papers/notes_limits_generalization_rlvr.md (16 pages, comprehensive)
- papers/notes_improving_factuality_debate.md (27 pages, comprehensive)
- papers/notes_gsm_symbolic.md (22 pages, comprehensive)
- papers/notes_rlvr_correct_reasoning.md (31 pages, comprehensive)
- papers/notes_additional_papers.md (6 papers skimmed)

---

## Datasets
Total datasets downloaded: **5**

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| GSM8K | openai/gsm8k | 7,473+1,319 | Grade school math | datasets/gsm8k/ | Standard math reasoning benchmark |
| MATH | EleutherAI/hendrycks_math | 7,500+5,000 | Competition math | datasets/math/ | 7 subject areas combined |
| GSM-Symbolic | apple/GSM-Symbolic | 5,000 | Robustness eval | datasets/gsm_symbolic/ | Symbolic variants for robustness testing |
| MATH-500 | HuggingFaceH4/MATH-500 | 500 | Math eval subset | datasets/math_500/ | Standard evaluation subset |
| DAPO-Math-17K | open-r1/DAPO-Math-17k-Processed | 17,398 | RLVR training | datasets/dapo_math_17k/ | Primary RLVR training dataset |

See datasets/README.md for download instructions and loading code.
Sample data in datasets/samples/ (JSON, first 5 examples each).

---

## Code Repositories
Total repositories cloned: **6**

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| verl | github.com/volcengine/verl | Core RL training framework | code/verl/ | Used by SDRL, SimpleRL-Zoo; supports GRPO, PPO, DAPO |
| OpenRLHF | github.com/OpenRLHF/OpenRLHF | Alternative RLHF/RLVR framework | code/OpenRLHF/ | Ray + vLLM + DeepSpeed |
| SimpleRL-Zoo | github.com/hkust-nlp/simpleRL-reason | Zero RL training recipes | code/SimpleRL-Zoo/ | 10 models, GRPO on GSM8K+MATH |
| open-r1 | github.com/huggingface/open-r1 | DeepSeek-R1 reproduction | code/open-r1/ | SFT distillation + GRPO pipelines |
| LLM-debate | github.com/composable-models/llm_multiagent_debate | Multi-agent debate impl | code/LLM-debate/ | Math, GSM, biography, MMLU tasks |
| GSM-Symbolic | github.com/apple/ml-gsm-symbolic | Symbolic GSM8K templates | code/GSM-Symbolic/ | Templates + generated data |

See code/README.md for detailed descriptions and key files.

---

## Resource Gathering Notes

### Search Strategy
1. Used arXiv API with targeted queries across 5 search themes (RLVR, collaborative reasoning, robustness, specific key papers, GRPO)
2. Attempted Semantic Scholar API (rate-limited, fell back to arXiv)
3. Paper-finder service was unavailable; conducted manual programmatic search
4. Identified 111 candidate papers, curated to 33 most relevant

### Selection Criteria
- Direct relevance to RLVR, multi-agent debate, or robustness of reasoning
- Recency (2023-2026 preferred)
- Availability of code/data
- Citation impact and foundational importance

### Challenges Encountered
- Semantic Scholar API rate-limited (429 errors)
- Paper-finder service not running
- 3 PDFs contained different papers than expected (arxiv ID mismatch with content)
- One paper (2405.13243v2) returned 404; alternative version found

### Gaps and Workarounds
- No single paper combines RLVR + collaborative training + robustness evaluation (our research gap)
- SDRL (2601.22297) is closest but doesn't evaluate on robustness benchmarks
- MACA (2509.15172) uses debate for self-improvement but doesn't use verifiable rewards

---

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **Training:** DAPO-Math-17K (standard, used by SDRL and RLVR correct reasoning paper)
- **Robustness evaluation:** GSM-Symbolic (from apple/GSM-Symbolic)
- **Standard evaluation:** MATH-500 + GSM8K test

### 2. Baseline Methods
- Standard GRPO/DAPO (no collaboration) — verl framework
- Inference-time multi-agent debate on RLVR models — LLM-debate code
- SDRL approach (joint single-agent + debate training) — adapt from SDRL paper methodology

### 3. Evaluation Metrics
- Pass@1, maj@K for standard accuracy
- Debate delta (Δ) for collaborative improvement
- GSM-Symbolic variance and NoOp accuracy for robustness
- CoT-Pass@K if CoT verifier is available

### 4. Code to Adapt/Reuse
- **verl** for GRPO/DAPO RL training infrastructure
- **LLM-debate** for multi-agent debate implementation
- **GSM-Symbolic** for robustness evaluation templates
- **SimpleRL-Zoo** for training recipes and hyperparameter guidance
- **open-r1** for reward implementations
