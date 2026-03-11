"""
Collaborative RLVR for Robust Reasoning: Inference-Time Experiment

Tests whether multi-agent debate improves robustness of mathematical reasoning,
measured by performance stability between GSM8K (standard) and GSM-Symbolic (distribution-shifted).

Conditions:
1. Single agent (greedy decoding)
2. Self-Consistency@3 (3 samples, majority vote)
3. Collaborative Debate (2 agents solve independently, share solutions, revise)
"""

import json
import random
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================================
# Configuration
# ============================================================================
SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"
MAX_NEW_TOKENS = 300  # Enough for full GSM8K solutions
NUM_PROBLEMS = 15  # Unique problems
NUM_SYMBOLIC_VARIANTS = 2  # Symbolic variants per problem
SC_K = 3
TEMPERATURE = 0.7
TOP_P = 0.9
RESULTS_DIR = Path("results")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(32)


# ============================================================================
# Answer Extraction
# ============================================================================
def extract_numeric_answer(text):
    """Extract final numeric answer from model output or ground truth."""
    if text is None:
        return None
    # Try boxed first (model's preferred format)
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return normalize_number(match.group(1))
    # Try "Answer: X" pattern
    match = re.search(r'Answer:\s*\$?\s*([-\d,./]+)', text)
    if match:
        return normalize_number(match.group(1))
    # Try #### pattern (GSM8K format)
    match = re.search(r'####\s*([-\d,./]+)', text)
    if match:
        return normalize_number(match.group(1))
    # Try "the answer is X"
    match = re.search(r'the answer is\s*\$?\s*([-\d,./]+)', text, re.IGNORECASE)
    if match:
        return normalize_number(match.group(1))
    # Fall back to last number in text
    numbers = re.findall(r'[-]?\d[\d,]*\.?\d*', text)
    if numbers:
        return normalize_number(numbers[-1])
    return None


def normalize_number(s):
    """Normalize a number string for comparison."""
    if s is None:
        return None
    s = s.strip().replace(',', '').replace('$', '').replace('%', '')
    s = s.rstrip('.')
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s


def answers_match(pred, gold):
    """Check if predicted answer matches gold answer."""
    if pred is None or gold is None:
        return False
    return pred.strip() == gold.strip()


# ============================================================================
# Data Preparation
# ============================================================================
def prepare_data():
    """Select problems and their symbolic variants."""
    print("Loading datasets...")
    gsm8k = load_from_disk("datasets/gsm8k/data")
    gsm_sym = load_from_disk("datasets/gsm_symbolic/data")

    gsm_sym_test = gsm_sym["test"]

    # Group GSM-Symbolic by original_id
    sym_by_id = {}
    for ex in gsm_sym_test:
        oid = ex["original_id"]
        if oid not in sym_by_id:
            sym_by_id[oid] = []
        sym_by_id[oid].append(ex)

    # Select problems
    selected_ids = random.sample(list(sym_by_id.keys()), min(NUM_PROBLEMS, len(sym_by_id)))

    problems = []
    for oid in selected_ids:
        variants = sym_by_id[oid]
        original = variants[0]

        # Original problem
        gold_answer = extract_numeric_answer(original["original_answer"])
        problems.append({
            "id": f"orig_{oid}",
            "question": original["original_question"],
            "gold_answer": gold_answer,
            "type": "original",
            "original_id": oid,
        })

        # Symbolic variants
        selected_variants = random.sample(variants, min(NUM_SYMBOLIC_VARIANTS, len(variants)))
        for v in selected_variants:
            gold_sym = extract_numeric_answer(v["answer"])
            problems.append({
                "id": f"sym_{oid}_inst{v['instance']}",
                "question": v["question"],
                "gold_answer": gold_sym,
                "type": "symbolic",
                "original_id": oid,
                "instance": v["instance"],
            })

    print(f"Selected {len(problems)} problems ({NUM_PROBLEMS} originals + "
          f"{len(problems)-NUM_PROBLEMS} symbolic variants)")
    return problems


# ============================================================================
# Model Loading and Generation
# ============================================================================
def load_model():
    """Load model with dynamic quantization for fast CPU inference."""
    print(f"Loading model {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.eval()

    # Apply dynamic quantization for ~4x CPU speedup
    print("Applying dynamic INT8 quantization...")
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    print(f"Model loaded and quantized in {time.time()-t0:.1f}s")
    return model, tokenizer


def generate_solution(model, tokenizer, question, system_prompt=None, do_sample=False, temp=0.7):
    """Generate a single solution for a math problem."""
    if system_prompt is None:
        system_prompt = "Solve step by step. Put final answer in \\boxed{}."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temp
        gen_kwargs["top_p"] = TOP_P

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def generate_debate_revision(model, tokenizer, question, own_solution, other_solution):
    """Generate a revised solution after seeing another agent's solution."""
    system_prompt = "Review both solutions. Find errors. Give correct answer in \\boxed{}."
    user_prompt = (
        f"Problem: {question}\n\n"
        f"Solution A:\n{own_solution}\n\n"
        f"Solution B:\n{other_solution}\n\n"
        f"Which is correct? Give the right answer."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.5,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


# ============================================================================
# Experimental Conditions
# ============================================================================
def run_single_agent(model, tokenizer, problems):
    """Condition 1: Single agent greedy decoding."""
    print("\n" + "="*60)
    print("CONDITION 1: Single Agent (Greedy)")
    print("="*60)
    results = []
    for i, prob in enumerate(problems):
        t0 = time.time()
        solution = generate_solution(model, tokenizer, prob["question"], do_sample=False)
        elapsed = time.time() - t0
        pred = extract_numeric_answer(solution)
        correct = answers_match(pred, prob["gold_answer"])
        results.append({
            "problem_id": prob["id"],
            "type": prob["type"],
            "original_id": prob["original_id"],
            "gold_answer": prob["gold_answer"],
            "predicted_answer": pred,
            "correct": correct,
            "solution": solution,
            "time_s": elapsed,
        })
        status = "OK" if correct else "WRONG"
        print(f"  [{i+1}/{len(problems)}] {prob['id']}: pred={pred}, gold={prob['gold_answer']} "
              f"[{status}] ({elapsed:.1f}s)")
        sys.stdout.flush()
    return results


def run_self_consistency(model, tokenizer, problems, k=SC_K):
    """Condition 2: Self-Consistency with majority voting."""
    print("\n" + "="*60)
    print(f"CONDITION 2: Self-Consistency@{k}")
    print("="*60)
    results = []
    for i, prob in enumerate(problems):
        t0 = time.time()
        answers = []
        solutions = []
        for j in range(k):
            sol = generate_solution(model, tokenizer, prob["question"], do_sample=True, temp=TEMPERATURE)
            ans = extract_numeric_answer(sol)
            answers.append(ans)
            solutions.append(sol)

        # Majority vote
        valid = [a for a in answers if a is not None]
        if valid:
            majority = Counter(valid).most_common(1)[0][0]
        else:
            majority = None

        elapsed = time.time() - t0
        correct = answers_match(majority, prob["gold_answer"])
        results.append({
            "problem_id": prob["id"],
            "type": prob["type"],
            "original_id": prob["original_id"],
            "gold_answer": prob["gold_answer"],
            "individual_answers": answers,
            "predicted_answer": majority,
            "correct": correct,
            "solutions": solutions,
            "time_s": elapsed,
        })
        status = "OK" if correct else "WRONG"
        print(f"  [{i+1}/{len(problems)}] {prob['id']}: votes={answers}, "
              f"majority={majority}, gold={prob['gold_answer']} [{status}] ({elapsed:.1f}s)")
        sys.stdout.flush()
    return results


def run_debate(model, tokenizer, problems):
    """Condition 3: Collaborative Debate (1 round)."""
    print("\n" + "="*60)
    print("CONDITION 3: Collaborative Debate (1 round)")
    print("="*60)
    results = []
    for i, prob in enumerate(problems):
        t0 = time.time()
        # Round 0: Two independent solutions
        sol_a = generate_solution(model, tokenizer, prob["question"], do_sample=True, temp=TEMPERATURE)
        sol_b = generate_solution(model, tokenizer, prob["question"], do_sample=True, temp=TEMPERATURE)
        ans_a = extract_numeric_answer(sol_a)
        ans_b = extract_numeric_answer(sol_b)

        # Round 1: Each agent revises after seeing the other's solution
        rev_a = generate_debate_revision(model, tokenizer, prob["question"], sol_a, sol_b)
        rev_b = generate_debate_revision(model, tokenizer, prob["question"], sol_b, sol_a)
        rev_ans_a = extract_numeric_answer(rev_a)
        rev_ans_b = extract_numeric_answer(rev_b)

        # Final answer: majority vote across all 4 answers
        all_answers = [ans_a, ans_b, rev_ans_a, rev_ans_b]
        valid = [a for a in all_answers if a is not None]
        if valid:
            majority = Counter(valid).most_common(1)[0][0]
        else:
            majority = None

        elapsed = time.time() - t0
        correct = answers_match(majority, prob["gold_answer"])
        results.append({
            "problem_id": prob["id"],
            "type": prob["type"],
            "original_id": prob["original_id"],
            "gold_answer": prob["gold_answer"],
            "initial_answers": [ans_a, ans_b],
            "revised_answers": [rev_ans_a, rev_ans_b],
            "all_answers": all_answers,
            "predicted_answer": majority,
            "correct": correct,
            "solutions": {
                "agent_a_initial": sol_a,
                "agent_b_initial": sol_b,
                "agent_a_revised": rev_a,
                "agent_b_revised": rev_b,
            },
            "time_s": elapsed,
        })
        status = "OK" if correct else "WRONG"
        print(f"  [{i+1}/{len(problems)}] {prob['id']}: "
              f"initial=[{ans_a},{ans_b}] revised=[{rev_ans_a},{rev_ans_b}] "
              f"majority={majority}, gold={prob['gold_answer']} [{status}] ({elapsed:.1f}s)")
        sys.stdout.flush()
    return results


# ============================================================================
# Main
# ============================================================================
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "seed": SEED,
        "model": MODEL_NAME,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_problems": NUM_PROBLEMS,
        "num_symbolic_variants": NUM_SYMBOLIC_VARIANTS,
        "sc_k": SC_K,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "quantization": "dynamic_int8",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Prepare data
    problems = prepare_data()
    with open(RESULTS_DIR / "problems.json", "w") as f:
        json.dump(problems, f, indent=2)

    # Load model
    model, tokenizer = load_model()

    # Run experiments
    total_start = time.time()

    # Condition 1: Single Agent
    single_results = run_single_agent(model, tokenizer, problems)
    with open(RESULTS_DIR / "single_agent_results.json", "w") as f:
        json.dump(single_results, f, indent=2, default=str)
    elapsed = time.time() - total_start
    n_correct = sum(r["correct"] for r in single_results)
    print(f"\nSingle agent: {n_correct}/{len(single_results)} correct. Elapsed: {elapsed:.0f}s")

    # Condition 2: Self-Consistency
    sc_results = run_self_consistency(model, tokenizer, problems)
    with open(RESULTS_DIR / "sc_results.json", "w") as f:
        json.dump(sc_results, f, indent=2, default=str)
    elapsed = time.time() - total_start
    n_correct = sum(r["correct"] for r in sc_results)
    print(f"\nSC@3: {n_correct}/{len(sc_results)} correct. Elapsed: {elapsed:.0f}s")

    # Condition 3: Debate
    debate_results = run_debate(model, tokenizer, problems)
    with open(RESULTS_DIR / "debate_results.json", "w") as f:
        json.dump(debate_results, f, indent=2, default=str)
    elapsed = time.time() - total_start
    n_correct = sum(r["correct"] for r in debate_results)
    print(f"\nDebate: {n_correct}/{len(debate_results)} correct. Elapsed: {elapsed:.0f}s")

    # Quick summary
    total_elapsed = time.time() - total_start
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)

    for name, res in [("Single", single_results), ("SC@3", sc_results), ("Debate", debate_results)]:
        orig = [r for r in res if r["type"] == "original"]
        sym = [r for r in res if r["type"] == "symbolic"]
        orig_acc = sum(r["correct"] for r in orig) / len(orig) if orig else 0
        sym_acc = sum(r["correct"] for r in sym) / len(sym) if sym else 0
        rr = sym_acc / orig_acc if orig_acc > 0 else float('nan')
        print(f"  {name:8s}: GSM8K={orig_acc:.1%}, GSM-Sym={sym_acc:.1%}, Robustness={rr:.3f}")

    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
