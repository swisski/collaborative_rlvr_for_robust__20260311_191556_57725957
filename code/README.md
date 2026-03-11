# Code Repositories for Collaborative RLVR for Robust Reasoning

This directory contains cloned reference repositories relevant to the research project.
All repos were cloned with `--depth 1` to save space.

---

## 1. verl (Volcano Engine Reinforcement Learning for LLMs)

- **URL:** https://github.com/volcengine/verl
- **Purpose:** Flexible, efficient, production-ready RL training framework for LLMs. The primary framework used in many RLVR papers including SimpleRL-Zoo, TinyZero, DAPO, and others. Supports GRPO, PPO, REINFORCE++, RLOO, DAPO, and many more algorithms. Built on FSDP/Megatron-LM for training and vLLM/SGLang for rollout generation.
- **Key files and directories:**
  - `verl/` - Core library code
  - `examples/grpo_trainer/` - GRPO training examples (most relevant for RLVR)
  - `examples/ppo_trainer/` - PPO training examples
  - `examples/reinforce_plus_plus_trainer/` - REINFORCE++ examples
  - `examples/data_preprocess/` - Data preparation utilities
  - `verl/experimental/` - Experimental features (transfer queue, async policy, VLA)
- **How to use:** Install via `pip install -e .`, launch Ray cluster, run training scripts from `examples/`. See https://verl.readthedocs.io for full docs.
- **Paper:** HybridFlow: A Flexible and Efficient RLHF Framework (arXiv:2409.19256)

---

## 2. OpenRLHF

- **URL:** https://github.com/OpenRLHF/OpenRLHF
- **Purpose:** Open-source, scalable RLHF/RLVR framework built on Ray + vLLM + DeepSpeed. First framework with unified agent-based execution paradigm. Supports PPO, REINFORCE++, REINFORCE++-baseline, GRPO, RLOO, Dr. GRPO. Includes both single-turn and multi-turn agent execution modes.
- **Key files and directories:**
  - `openrlhf/` - Core library
  - `openrlhf/cli/train_ppo_ray.py` - Main RL training entry point
  - `openrlhf/cli/train_sft.py` - SFT training
  - `openrlhf/cli/train_rm.py` - Reward model training
  - `examples/scripts/` - Ready-to-run training scripts
  - `examples/scripts/train_reinforce_baseline_hybrid_engine.sh` - REINFORCE++-baseline (best for RLVR)
  - `examples/scripts/train_ppo_with_reward_fn.sh` - Custom reward function example
  - `examples/scripts/train_reinforce_baseline_ray_agent_async.sh` - Multi-turn agent async training
- **How to use:** `pip install openrlhf[vllm]`, launch Ray, submit jobs via `ray job submit`. Use `--advantage_estimator reinforce_baseline` for RLVR tasks.
- **Paper:** OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework (arXiv:2405.11143); REINFORCE++ (arXiv:2501.03262)

---

## 3. SimpleRL-Zoo (simpleRL-reason)

- **URL:** https://github.com/hkust-nlp/simpleRL-reason
- **Purpose:** Simple zero RL training recipe for improving reasoning in base LLMs. Demonstrates that rule-based rewards + GSM8K/MATH data can boost accuracy by 10-20+ absolute points across 10 diverse models (0.5B to 32B). Uses GRPO via the verl framework. Key research finding: increased response length does not necessarily correlate with self-verification behavior.
- **Key files and directories:**
  - `train_grpo_math_tune_ray.sh` - Main training script (GRPO with Ray/vLLM)
  - `eval_math_nodes.sh` - Full evaluation pipeline
  - `launch_gradio.sh` - Visualization tool for comparing responses across training steps
  - `verl/` - Modified verl library code
  - `scripts/` - Utility scripts
  - `examples/` - Additional examples
- **How to use:** Install verl environment, download dataset from HuggingFace (`hkust-nlp/SimpleRL-Zoo-Data`), launch Ray cluster, run `train_grpo_math_tune_ray.sh` with model-specific parameters.
- **Paper:** SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild (arXiv:2503.18892)

---

## 4. open-r1

- **URL:** https://github.com/huggingface/open-r1
- **Purpose:** Fully open reproduction of DeepSeek-R1. Provides SFT distillation and GRPO training pipelines. Released Mixture-of-Thoughts dataset (350k verified reasoning traces), OpenR1-Math-220k, and CodeForces-CoTs. Uses TRL's vLLM backend for GRPO and lighteval for evaluation.
- **Key files and directories:**
  - `src/open_r1/grpo.py` - GRPO training script
  - `src/open_r1/sft.py` - SFT distillation script
  - `src/open_r1/generate.py` - Synthetic data generation via Distilabel
  - `src/open_r1/rewards.py` - Reward function implementations
  - `src/open_r1/configs.py` - Configuration definitions
  - `recipes/` - Training recipe configs (YAML) for various models
  - `scripts/decontaminate.py` - Data decontamination using 8-grams
  - `scripts/pass_rate_filtering/` - Dataset filtering by pass rate
  - `Makefile` - Easy-to-run pipeline commands
- **How to use:** Install via `uv pip install -e ".[dev]"`, use `accelerate launch` with recipe configs. For GRPO: `src/open_r1/grpo.py --config recipes/.../config.yaml`.
- **Citation:** Open R1 (Hugging Face, 2025)

---

## 5. LLM-debate (Multi-Agent Debate)

- **URL:** https://github.com/composable-models/llm_multiagent_debate
- **Purpose:** Implementation of "Improving Factuality and Reasoning in Language Models through Multiagent Debate." Multiple LLM agents debate to improve answers on math, GSM, biographies, and MMLU tasks. Directly relevant to the collaborative aspect of our research -- demonstrates that multi-agent interaction can improve reasoning without additional training.
- **Key files and directories:**
  - `math/gen_math.py` - Generate and evaluate math answers via debate
  - `gsm/gen_gsm.py` - Generate GSM answers via debate
  - `gsm/eval_gsm.py` - Evaluate GSM debate results
  - `biography/gen_conversation.py` - Biography generation via debate
  - `mmlu/gen_mmlu.py` - MMLU via debate
  - `requirements.txt` - Dependencies
- **How to use:** `cd` into task directory (math/, gsm/, biography/, mmlu/), run `python gen_<task>.py` to generate debate answers, then `python eval_<task>.py` to evaluate.
- **Paper:** Improving Factuality and Reasoning in Language Models through Multiagent Debate (arXiv:2305.14325)

---

## 6. GSM-Symbolic

- **URL:** https://github.com/apple/ml-gsm-symbolic
- **Purpose:** Apple's codebase for generating symbolic variants of GSM8K problems. Demonstrates limitations of LLM mathematical reasoning by testing on structurally identical but numerically different problems. Provides templates and generated data for GSM-Symbolic, GSM-Symbolic-P1 (added clauses), and GSM-Symbolic-P2 (removed clauses). Useful for creating robustness evaluation benchmarks.
- **Key files and directories:**
  - `templates/` - Original templates for generating symbolic variants
  - `generated_data/GSM_symbolic.jsonl` - 50 instances per template for GSM-Symbolic
  - `generated_data/` - Also contains P1 and P2 variants
- **How to use:** Use templates to generate new symbolic variants, or use pre-generated data in `generated_data/` directly. Each entry has `question`, `answer`, `original_id`, and `original_question` fields. Evaluation uses greedy decoding with 8-shot CoT prompting.
- **Paper:** GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models (arXiv:2410.05229)

---

## Relevance Summary

| Repository | Relevance to Collaborative RLVR |
|---|---|
| **verl** | Core RL training infrastructure; GRPO/PPO implementation for RLVR |
| **OpenRLHF** | Alternative RL framework; REINFORCE++-baseline for RLVR; multi-turn agent support |
| **SimpleRL-Zoo** | Proven zero RL recipe; findings on reasoning behavior emergence; built on verl |
| **open-r1** | R1 reproduction; GRPO recipes; evaluation pipelines; distillation datasets |
| **LLM-debate** | Multi-agent debate for collaborative reasoning (inference-time collaboration) |
| **GSM-Symbolic** | Robustness evaluation via symbolic math variants; tests reasoning generalization |
