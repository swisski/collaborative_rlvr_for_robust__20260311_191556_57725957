# Downloaded Papers

33 papers organized by research theme. Deep-read notes available for 6 key papers + skimmed notes for 6 additional papers.

## RLVR Core

1. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** (2501.12948v1_deepseek_r1.pdf)
   - Authors: DeepSeek-AI | Year: 2025
   - arXiv: 2501.12948 | Deep-read notes: notes_deepseek_r1.md
   - Why relevant: Foundational RLVR paper. GRPO algorithm, 4-stage pipeline, rule-based rewards.

2. **Reasoning Gym** (2505.24760v2_reasoning_gym.pdf)
   - Year: 2025 | arXiv: 2505.24760
   - Why relevant: RL environments designed for verifiable rewards.

3. **RLVR Implicitly Incentivizes Correct Reasoning** (2506.14245v2_rlvr_correct_reasoning.pdf)
   - Authors: Wen et al. | Year: 2025
   - arXiv: 2506.14245 | Deep-read notes: notes_rlvr_correct_reasoning.md
   - Why relevant: Proves RLVR improves reasoning quality via Logic Prior. Novel CoT-Pass@K metric.

4. **RLVR with Noisy Rewards** (2510.00915v3_rlvr_noisy_rewards.pdf)
   - Year: 2025 | arXiv: 2510.00915
   - Why relevant: Handling imperfect verifiers in RLVR.

5. **Trust But Verify** (2505.13445v1_trust_but_verify_rlvr.pdf)
   - Year: 2025 | arXiv: 2505.13445
   - Why relevant: Self-verification approach complementary to collaborative verification.

6. **Limits of Generalization in RLVR** (2510.27044v2_limits_generalization_rlvr.pdf)
   - Authors: Alam & Rastogi | Year: 2025
   - arXiv: 2510.27044 | Deep-read notes: notes_limits_generalization_rlvr.md
   - Why relevant: Core motivation paper. Shows RLVR amplifies heuristics, not algorithms. Recommends sequence-aware rewards.

7. **SimpleRL-Zoo** (2503.18892v3_simplerl_zoo.pdf)
   - Authors: Zeng et al. | Year: 2025
   - arXiv: 2503.18892 | Skimmed notes: notes_additional_papers.md
   - Why relevant: Systematic zero RL study. Training recipes for diverse models.

8. **AceReason-Nemotron** (2505.16400v3_acereason_nemotron.pdf)
   - Year: 2025 | arXiv: 2505.16400
   - Why relevant: Math+code reasoning advancement via RL.

9. **Dual-Token Constraints for RLVR** (2507.15778v1_dual_token_rlvr.pdf)
   - Year: 2025 | arXiv: 2507.15778
   - Why relevant: Stabilizing knowledge during RLVR training.

10. **Uncertainty-Aware Advantage Shaping** (2510.10649v1_uncertainty_rlvr.pdf)
    - Year: 2025 | arXiv: 2510.10649
    - Why relevant: Exploration improvement via uncertainty estimation.

11. **Med-RLVR** (2502.19655v1_med_rlvr.pdf)
    - Year: 2025 | arXiv: 2502.19655
    - Why relevant: RLVR applied to medical domain from small (3B) model.

12. **RL for Reasoning with One Example** (2504.20571v3_rl_one_example.pdf)
    - Year: 2025 | arXiv: 2504.20571
    - Why relevant: Questions whether RLVR induces new reasoning.

13. **Making Mathematical Reasoning Adaptive** (2510.04617v2_adaptive_math_reasoning.pdf)
    - Year: 2025 | arXiv: 2510.04617
    - Why relevant: Adaptive reasoning approaches for math.

## Multi-Agent Debate

14. **Self-Debate RL (SDRL)** (2601.22297v1_self_debate_rl.pdf)
    - Authors: Liu et al. | Year: 2026
    - arXiv: 2601.22297 | Deep-read notes: notes_self_debate_rl.md
    - Why relevant: **Most relevant paper.** Joint RLVR + debate training via self-debate.

15. **Improving Factuality via Multiagent Debate** (2305.14325v1_improving_factuality_debate_du.pdf)
    - Authors: Du et al. (MIT, Google Brain) | Year: 2023
    - arXiv: 2305.14325 | Deep-read notes: notes_improving_factuality_debate.md
    - Why relevant: Foundational debate paper. GSM8K 77→85%, arithmetic 67→82%.

16. **Debate Only When Necessary (DOWN)** (2504.05047v2_debate_only_when_necessary.pdf)
    - Authors: Eo et al. | Year: 2025
    - arXiv: 2504.05047 | Skimmed notes: notes_additional_papers.md
    - Why relevant: Adaptive/selective debate reduces cost while maintaining accuracy.

17. **Debate or Vote** (2508.17536v2_debate_or_vote.pdf)
    - Year: 2025 | arXiv: 2508.17536
    - Why relevant: Comparative analysis of debate vs. voting strategies.

18. **Inter-Consistency via Debate** (2305.11595v3_inter_consistency_debate.pdf)
    - Year: 2023 | arXiv: 2305.11595
    - Why relevant: In-depth analysis of LLM collaboration dynamics.

19. **Corex: Multi-Model Collaboration** (2310.00280v3_corex_multi_model.pdf)
    - Year: 2023 | arXiv: 2310.00280
    - Why relevant: Pushing reasoning boundaries through multi-model collaboration.

20. **MARS: Efficient Multi-Agent Collaboration** (2509.20502v1_mars_multi_agent.pdf)
    - Year: 2025 | arXiv: 2509.20502
    - Why relevant: Efficiency improvements for multi-agent LLM reasoning.

21. **Self-Improvement via Debate (MACA)** (2509.15172v3_self_improvement_debate.pdf)
    - Authors: Samanta et al. | Year: 2026
    - arXiv: 2509.15172 | Skimmed notes: notes_additional_papers.md
    - Why relevant: DPO/KTO on debate traces for self-improvement. Up to +26.87% MATH.

22. **DynaDebate** (2601.05746v1_dynadebate.pdf)
    - Year: 2026 | arXiv: 2601.05746
    - Why relevant: Breaking homogeneity in multi-agent debate.

## Robustness

23. **GSM-Symbolic** (2410.05229v1_gsm_symbolic.pdf)
    - Authors: Mirzadeh et al. (Apple) | Year: 2024
    - arXiv: 2410.05229 | Deep-read notes: notes_gsm_symbolic.md
    - Why relevant: Reveals reasoning fragility. GSM-NoOp causes up to 65% drops. Evaluation methodology.

24. **Robust Reasoning with Noisy Rationales** (2410.23856v1_robust_reasoning_noisy.pdf)
    - Year: 2024 | arXiv: 2410.23856
    - Why relevant: Robustness under corrupted chain-of-thought.

25. **R3 Prompting** (2310.16535v1_r3_prompting.pdf)
    - Year: 2023 | arXiv: 2310.16535
    - Why relevant: Review, Rephrase, Resolve approach for CoT reasoning.

## Faithful Reasoning

26-31. Six papers on CoT faithfulness (see file list above).

## GRPO/RL Techniques

32-33. Stable RL and S-GRPO papers (see file list above).
