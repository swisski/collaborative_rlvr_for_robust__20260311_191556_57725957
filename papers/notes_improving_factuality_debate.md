# Improving Factuality and Reasoning in Language Models through Multiagent Debate

**Paper:** Du et al. (2023), arXiv:2305.14325v1
**Authors:** Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum (MIT CSAIL), Igor Mordatch (Google Brain)
**Date:** May 23, 2023

---

## 1. Core Idea: The Multi-Agent Debate Framework

### How Models Debate

The framework is inspired by Minsky's "The Society of Mind." Multiple LLM instances (agents) independently generate answers to a query, then iteratively critique and refine each other's responses over multiple rounds until they converge on a common final answer.

**Step-by-step procedure:**

1. **Round 1 (Independent generation):** Each of N agents independently generates a candidate answer to the given query. Each response represents a possible thought process or source of information.
2. **Debate rounds:** After initial responses, all agents' responses are concatenated and given as context to each agent via a "consensus prompt." Each agent is instructed to construct a new response considering other agents' answers.
3. **Iteration:** The consensus prompt is repeatedly applied using updated responses from each agent across multiple rounds.
4. **Final answer:** After the specified number of debate rounds, the agents' (now typically converged) answer is taken as the final output.

**Key design choice:** The approach requires only **black-box access** to language model generations -- no model-internal information (likelihoods, gradients) is needed. The same methodology and prompt templates are used for all tasks.

### Consensus Prompts

Two prompt variants control debate duration:

| Debate Length | Prompt |
|---|---|
| **Short** | "These are the solutions to the problem from other agents: [other answers]. Based off the opinion of other agents, can you give an updated response..." |
| **Long** | "These are the solutions to the problem from other agents: [other answers]. Using the opinion of other agents as additional advice, can you give an updated response..." |

The "long" prompt (which frames other answers as "additional advice" rather than authoritative opinions) encourages agents to be more "stubborn" about their own solutions, leading to longer debates and better final solutions.

### Convergence to Consensus

- Debate can theoretically be seen as a multi-agent game where convergence is NOT guaranteed.
- **Empirically**, language models almost always converge on a single shared answer after multiple rounds.
- Language model agents are relatively "agreeable," possibly due to instruction tuning or RLHF.
- Convergence speed is controllable via prompt design (short vs. long prompts).
- Short debate prompts induce faster consensus (~85% consensus by round 2); long prompts reach ~95% consensus by round 4.

---

## 2. How Debate Improves Factuality and Reasoning

### Mechanisms of Improvement

1. **Cross-verification of reasoning chains:** Multiple agents explore different solution paths. When agents examine each other's reasoning, errors in individual chains are identified and corrected.
2. **Not just majority voting -- genuine error correction:** Critically, debate does NOT just amplify one correct answer in a quorum. The paper shows many cases where **ALL models initially give incorrect answers**, but through the debate process, agents critique each other's reasoning and arrive at the correct answer (Figures 4, 11). This is a key distinction from simple majority voting.
3. **Uncertainty-driven fact filtering:** For factual tasks (biographies), different agents tend to hallucinate different false facts. During debate, agents disagree on uncertain facts and either correct or omit them. Consistent facts across agents are retained.
4. **"Ease of persuasion" as confidence signal:** When the model is confident about a fact (all instances give the same answer), it is very difficult to convince an agent to change its opinion. When uncertain, agents quickly change to a consensus -- suggesting persuadability correlates with factual uncertainty.

### Compatibility with Other Methods

Debate operates **orthogonally** to other prompting methods:
- Combined with zero-shot chain-of-thought (CoT) prompting on GSM8K: Single Agent without CoT = 74%, with CoT = 77%; Multi-Agent Debate without CoT = 77%, with CoT = **85%**.
- The gains from debate and CoT are additive/complementary.

---

## 3. Number of Rounds, Agents, and Convergence

### Default Configuration
- **Primary experiments:** 3 agents, 2 rounds of debate
- **Model:** gpt-3.5-turbo-0301 (ChatGPT) for all main experiments

### Scaling Agents (Figure 10a -- Arithmetic task)
- 1 agent: ~68% accuracy
- 2 agents: ~75%
- 3 agents: ~82%
- 5 agents: ~88%
- 7 agents: ~93%
- Performance **monotonically increases** with more agents.
- For larger numbers of agents (5+), responses are first summarized by ChatGPT to avoid context length issues (rather than direct concatenation).

### Scaling Rounds (Figure 10b -- Arithmetic task)
- 1 round: ~68% accuracy
- 2 rounds: ~82%
- 3 rounds: ~88%
- 4 rounds: ~90%
- Performance monotonically increases with debate length.
- **Diminishing returns above 4 rounds** -- additional rounds yield similar performance to 4 rounds.

### Consensus Rates (Figure 14)
- Short debate prompt: ~85% consensus at round 2, ~90% at round 4
- Long debate prompt: ~75% consensus at round 2, ~95% at round 4
- Long prompts: slower convergence but higher final accuracy.

---

## 4. Mathematical Reasoning Experiments and Results

### Tasks

1. **Arithmetic:** Evaluate expressions with six two-digit numbers using addition, multiplication, and subtraction (e.g., "What is the result of 12+15*21+0-3*27?"). 100 generated tasks.
2. **GSM8K (Grade School Math):** Grade school mathematical reasoning problems from the GSM8K dataset. 100 problems evaluated.
3. **Chess Move Prediction:** Predict best next move at turn 14 in grandmaster chess games (PGN notation). Evaluated by Stockfish pawn score with search depth 20. 300 chess games.

### Results (Table 1 -- Reasoning tasks, 3 agents, 2 rounds)

| Model | Arithmetic (%) | Grade School Math (%) | Chess (delta PS) |
|---|---|---|---|
| Single Agent | 67.0 +/- 4.7 | 77.0 +/- 4.2 | 91.4 +/- 10.6 |
| Single Agent (Reflection) | 72.1 +/- 4.5 | 75.0 +/- 4.3 | 102.1 +/- 11.9 |
| Multi-Agent (Majority Vote) | 69.0 +/- 4.6 | 81.0 +/- 3.9 | 102.2 +/- 6.2 |
| **Multi-Agent (Debate)** | **81.8 +/- 2.3** | **85.0 +/- 3.5** | **122.9 +/- 7.6** |

**Key findings on math reasoning:**
- Debate provides the largest improvement on arithmetic: +14.8 percentage points over single agent.
- On GSM8K: +8 percentage points over single agent.
- Debate substantially outperforms both self-reflection and majority voting.
- Self-reflection gives only modest gains on reasoning tasks (and sometimes hurts on GSM8K).
- Majority voting helps but is significantly weaker than debate.

---

## 5. Datasets and Benchmarks

### Six Benchmarks Evaluated

**Reasoning tasks (Table 1):**
1. **Arithmetic** -- Custom-generated expressions with 6 random integers (0-30), 100 tasks
2. **GSM8K** -- Grade school math word problems, 100 problems
3. **Chess Move Prediction** -- Best next move prediction, scored by Stockfish pawn advantage, 300 games

**Factuality tasks (Table 2):**
4. **Biographies** -- Novel benchmark: bullet-point biographies of 524 well-known computer scientists, compared against ground truth from Wikipedia. Evaluation uses ChatGPT to verify fact consistency.
5. **MMLU** -- Massive Multitask Language Understanding, 100 questions randomly distributed across subject areas
6. **Chess Move Validity** -- BIG-Bench Chess-State Tracking Benchmark (synthetic_short), predicting valid chess moves, 100 tasks

### Factuality Results (Table 2 -- 3 agents, 2 rounds)

| Model | Biographies | MMLU | Chess Move Validity |
|---|---|---|---|
| Single Agent | 66.0 +/- 2.2 | 63.9 +/- 4.8 | 29.3 +/- 2.6 |
| Single Agent (Reflection) | 68.3 +/- 2.9 | 57.7 +/- 5.0 | 38.8 +/- 2.9 |
| **Multi-Agent (Debate)** | **73.8 +/- 2.3** | **71.1 +/- 4.6** | **45.2 +/- 2.9** |

**Key findings on factuality:**
- Reflection actually **hurts** MMLU performance (63.9 -> 57.7), but debate helps substantially (63.9 -> 71.1).
- Chess move validity sees massive improvement: 29.3 -> 45.2 (+15.9 points).
- Biographies improve from 66.0 to 73.8 (+7.8 points).

### Summary Figure (Figure 1 -- All six benchmarks)

| Benchmark | Single Model | Multi-Model Debate |
|---|---|---|
| Biographies | 66 | **74** |
| MMLU | 64 | **71** |
| Chess Move Validity | 29 | **45** |
| Arithmetic | 67 | **82** |
| Grade School Math | 77 | **85** |
| Chess Move Optimality | 74 | **100** |

---

## 6. Comparison to Single-Model Approaches

### Baselines Compared
1. **Single Agent:** Direct generation of responses.
2. **Single Agent (Reflection):** Generate then self-critique/refine (a la Reflexion, Self-Refine).
3. **Multi-Agent (Majority Vote):** Generate with multiple agents, take majority answer.
4. **Multi-Agent (Debate):** The proposed approach.

### Key Comparative Findings

- **Debate vs. Single Agent:** Consistent substantial improvements across all 6 tasks (+7.2 to +15.9 percentage points).
- **Debate vs. Reflection:** Reflection gives modest boosts for reasoning but can HURT factuality (MMLU drops from 63.9 to 57.7 with reflection). Debate never hurts and always provides significant gains.
- **Debate vs. Majority Voting:** On reasoning tasks where majority voting is applicable, debate significantly outperforms it (Arithmetic: 69.0 vs 81.8; GSM8K: 81.0 vs 85.0). Majority voting is not directly applicable to open-ended factuality tasks.
- **Debate = Reflection + Multi-agent synergy:** Debate can be seen as combining the benefits of reflection (self-critique) with multi-agent generation, achieving more than either alone.

---

## 7. Analysis: When Debate Helps vs. Doesn't

### When Debate Helps Most

1. **Tasks where models generate diverse initial answers:** Debate works because different agents explore different reasoning paths. The diversity of initial answers is key.
2. **Tasks with verifiable reasoning chains:** Arithmetic, GSM8K -- agents can check each other's step-by-step work.
3. **Tasks where the model is uncertain:** Different agents give different answers on uncertain facts; debate converges to the more accurate consensus.
4. **Cross-model debate:** Testing chatGPT vs Bard on 20 GSM8K problems: Bard solves 11, ChatGPT solves 14, joint debate solves **17** -- debate improves both models.

### When Debate Has Limitations / Doesn't Help

1. **When all agents are confidently wrong about the same thing:** If all models converge on the same incorrect answer initially, debate may reinforce the error. Agents confidently affirm incorrect answers.
2. **Very long debates with current models:** As debates grow longer, models struggle to process the entire debate history and tend to focus only on most recent generations.
3. **Cases where initial incorrect consensus is strong:** When models converge on an incorrect answer early, they tend to confidently affirm it is correct and consistent with all other agents, making correction difficult.
4. **Factual knowledge limitations:** Debate can filter hallucinations but cannot inject new knowledge that no agent possesses.

### Failure Examples

The paper includes examples of incorrect GSM8K debates (Figures 21-23) where:
- All agents initially get the wrong answer and converge on an incorrect consensus.
- Example: The "concert attendance" problem where the correct agent (1.2x = 48, x = 40) is persuaded to adopt the incorrect approach (48 - 0.20*48 = 38.4).
- Example: A toy manufacturing problem where agents keep changing their answers across rounds without converging correctly.

---

## 8. Cost Analysis (Compute Overhead)

### Direct Cost

- Debate requires **N agents x R rounds** of LLM generations, compared to 1 generation for single agent.
- Default configuration (3 agents, 2 rounds) = **~6x the compute** of a single agent query (3 initial + 3 debate round responses, though debate rounds have longer context).
- Context length grows with each round as all agents' responses are concatenated.

### Mitigation Strategies

1. **Summarization:** For large numbers of agents, responses can be summarized before being passed as context. This actually **improves** performance (Figure 13) while reducing context length.
2. **Limited rounds:** Diminishing returns after 4 rounds -- 2-3 rounds capture most of the benefit.

### Cost-Benefit Perspective

- The authors argue debate can be used to **generate training data** that is distilled back into the base model, creating a self-improvement loop.
- Higher quality outputs from debate may justify the compute cost for critical applications.
- The approach is orthogonal to efficiency techniques and can be combined with retrieval, prompt engineering, etc.

---

## 9. Additional Design Insights

### Different Initialization Prompts
- Using different personas (professor, doctor, mathematician) for each agent improved MMLU from 71.1 to **74.2**, suggesting diversity in agent initialization further boosts performance.

### Summarization of Responses
- When many agents participate, summarizing all other agents' responses into a single response (rather than concatenating) before providing context to each agent actually **improves** performance while reducing context length.
- This enables scaling to 5+ agents without context length issues.

### Model Used
- All main experiments use `gpt-3.5-turbo-0301`.
- Zero-shot setting throughout (no few-shot examples).
- Zero-shot chain-of-thought applied in evaluations.

---

## 10. Key Takeaways for Our Research

1. **Multi-agent debate is a training-free method** that improves both reasoning and factuality using only black-box model access -- relevant for understanding how collaborative verification can emerge.
2. **Debate goes beyond majority voting:** The process of mutual critique and refinement creates genuinely new reasoning that no single agent initially produced.
3. **Scalability:** More agents and more rounds monotonically improve performance (with diminishing returns after ~4 rounds and ~7 agents on arithmetic).
4. **Complementary to other methods:** Debate stacks with chain-of-thought and other prompting techniques.
5. **Diversity matters:** Using different initialization prompts (personas) further improves debate outcomes.
6. **Cost is the primary limitation:** N agents x R rounds multiplier on compute. But summarization and limited rounds can keep this manageable.
7. **Convergence to incorrect answers remains a failure mode:** When models are confidently wrong about the same thing, debate reinforces errors rather than correcting them.
8. **"Ease of persuasion" as an uncertainty proxy:** How easily agents change their minds during debate correlates with factual uncertainty -- a useful signal.

---

## 11. Relevance to Collaborative RLVR

- This paper demonstrates that **collaborative multi-agent interaction improves reasoning** without any training, purely through inference-time debate.
- For RLVR, the debate mechanism could serve as: (a) a verification signal during training, (b) a way to generate higher-quality training data via self-improvement loops, (c) a complementary inference-time technique on top of RLVR-trained models.
- The finding that debate can correct errors even when ALL agents are initially wrong suggests that the collaborative process itself has emergent capabilities beyond individual model capacity.
- The cost analysis is relevant: RLVR training could potentially internalize the benefits of debate, producing models that achieve debate-level performance at single-model inference cost.
