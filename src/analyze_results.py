"""
Analyze experimental results: accuracy, robustness, statistical tests, and visualizations.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def load_results():
    """Load all experimental results."""
    results = {}
    for name, fname in [
        ("single", "single_agent_results.json"),
        ("sc3", "sc_results.json"),
        ("debate", "debate_results.json"),
    ]:
        path = RESULTS_DIR / fname
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
    return results


def compute_accuracy(results, problem_type=None):
    """Compute accuracy for a set of results, optionally filtered by type."""
    if problem_type:
        results = [r for r in results if r["type"] == problem_type]
    if not results:
        return 0.0, 0, []
    correct = [r["correct"] for r in results]
    return np.mean(correct), len(correct), correct


def bootstrap_ci(data, n_boot=10000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for a proportion."""
    rng = np.random.RandomState(seed)
    data = np.array(data, dtype=float)
    n = len(data)
    if n == 0:
        return 0.0, 0.0
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.sort(boot_means)
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return lo, hi


def mcnemar_test(results_a, results_b):
    """McNemar's test for paired binary outcomes."""
    assert len(results_a) == len(results_b)
    # Build contingency
    b_only = 0  # A wrong, B right
    a_only = 0  # A right, B wrong
    for ra, rb in zip(results_a, results_b):
        if ra["correct"] and not rb["correct"]:
            a_only += 1
        elif not ra["correct"] and rb["correct"]:
            b_only += 1
    # McNemar's with continuity correction
    n = b_only + a_only
    if n == 0:
        return 1.0, 0, 0  # No discordant pairs
    chi2 = (abs(b_only - a_only) - 1)**2 / (b_only + a_only)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return p_value, a_only, b_only


def cohens_h(p1, p2):
    """Compute Cohen's h effect size for two proportions."""
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def analyze_robustness(all_results):
    """Compute robustness metrics for each condition."""
    print("\n" + "="*70)
    print("ROBUSTNESS ANALYSIS")
    print("="*70)

    summary = {}
    for cond_name, results in all_results.items():
        orig_acc, n_orig, orig_correct = compute_accuracy(results, "original")
        sym_acc, n_sym, sym_correct = compute_accuracy(results, "symbolic")

        orig_ci = bootstrap_ci(orig_correct)
        sym_ci = bootstrap_ci(sym_correct)

        rr = sym_acc / orig_acc if orig_acc > 0 else float("nan")
        delta = orig_acc - sym_acc  # Drop in accuracy

        summary[cond_name] = {
            "orig_acc": orig_acc,
            "orig_n": n_orig,
            "orig_ci": orig_ci,
            "sym_acc": sym_acc,
            "sym_n": n_sym,
            "sym_ci": sym_ci,
            "robustness_ratio": rr,
            "accuracy_drop": delta,
        }

        print(f"\n{cond_name.upper()}:")
        print(f"  GSM8K Original: {orig_acc:.1%} (n={n_orig}) "
              f"[95% CI: {orig_ci[0]:.1%}, {orig_ci[1]:.1%}]")
        print(f"  GSM-Symbolic:   {sym_acc:.1%} (n={n_sym}) "
              f"[95% CI: {sym_ci[0]:.1%}, {sym_ci[1]:.1%}]")
        print(f"  Robustness Ratio: {rr:.3f}")
        print(f"  Accuracy Drop:    {delta:.1%}")

    return summary


def analyze_debate_dynamics(all_results):
    """Analyze how debate changes answers (correct→wrong, wrong→correct)."""
    if "debate" not in all_results:
        return {}

    print("\n" + "="*70)
    print("DEBATE DYNAMICS ANALYSIS")
    print("="*70)

    debate_results = all_results["debate"]
    dynamics = {"total": 0, "initial_agree": 0, "initial_disagree": 0,
                "flip_to_correct": 0, "flip_to_wrong": 0, "stayed_correct": 0,
                "stayed_wrong": 0}

    for r in debate_results:
        dynamics["total"] += 1
        init_answers = r.get("initial_answers", [])
        if len(init_answers) == 2 and init_answers[0] == init_answers[1]:
            dynamics["initial_agree"] += 1
        else:
            dynamics["initial_disagree"] += 1

        # Check if initial majority was correct
        from collections import Counter
        init_valid = [a for a in init_answers if a is not None]
        if init_valid:
            init_majority = Counter(init_valid).most_common(1)[0][0]
        else:
            init_majority = None
        init_correct = (init_majority == r["gold_answer"]) if init_majority else False

        final_correct = r["correct"]

        if init_correct and final_correct:
            dynamics["stayed_correct"] += 1
        elif init_correct and not final_correct:
            dynamics["flip_to_wrong"] += 1
        elif not init_correct and final_correct:
            dynamics["flip_to_correct"] += 1
        else:
            dynamics["stayed_wrong"] += 1

    print(f"  Total problems: {dynamics['total']}")
    print(f"  Initial agreement: {dynamics['initial_agree']} "
          f"({dynamics['initial_agree']/max(dynamics['total'],1):.1%})")
    print(f"  Initial disagreement: {dynamics['initial_disagree']} "
          f"({dynamics['initial_disagree']/max(dynamics['total'],1):.1%})")
    print(f"  Stayed correct:      {dynamics['stayed_correct']}")
    print(f"  Flipped to correct:  {dynamics['flip_to_correct']}")
    print(f"  Flipped to wrong:    {dynamics['flip_to_wrong']}")
    print(f"  Stayed wrong:        {dynamics['stayed_wrong']}")

    return dynamics


def statistical_tests(all_results, summary):
    """Run statistical tests comparing conditions."""
    print("\n" + "="*70)
    print("STATISTICAL TESTS")
    print("="*70)

    test_results = {}
    conditions = list(all_results.keys())

    # Pairwise McNemar tests
    for i in range(len(conditions)):
        for j in range(i+1, len(conditions)):
            ca, cb = conditions[i], conditions[j]
            # Overall
            p_val, a_only, b_only = mcnemar_test(all_results[ca], all_results[cb])
            h = cohens_h(summary[ca]["orig_acc"], summary[cb]["orig_acc"])
            key = f"{ca}_vs_{cb}"
            test_results[key] = {
                "p_value": p_val,
                "a_only": a_only,
                "b_only": b_only,
                "cohens_h": h,
            }
            sig = "*" if p_val < 0.05 else "ns"
            print(f"\n  {ca} vs {cb} (overall):")
            print(f"    McNemar p={p_val:.4f} {sig}")
            print(f"    {ca} correct, {cb} wrong: {a_only}")
            print(f"    {cb} correct, {ca} wrong: {b_only}")
            print(f"    Cohen's h: {h:.3f}")

            # By type
            for ptype in ["original", "symbolic"]:
                ra_typed = [r for r in all_results[ca] if r["type"] == ptype]
                rb_typed = [r for r in all_results[cb] if r["type"] == ptype]
                if ra_typed and rb_typed:
                    p_val, a_only, b_only = mcnemar_test(ra_typed, rb_typed)
                    sig = "*" if p_val < 0.05 else "ns"
                    print(f"    {ptype}: McNemar p={p_val:.4f} {sig} "
                          f"(discordant: {ca}={a_only}, {cb}={b_only})")

    return test_results


def plot_accuracy_comparison(summary):
    """Bar chart comparing accuracy across conditions and problem types."""
    conditions = list(summary.keys())
    labels = {"single": "Single Agent", "sc3": "SC@3", "debate": "Debate"}

    orig_accs = [summary[c]["orig_acc"] for c in conditions]
    sym_accs = [summary[c]["sym_acc"] for c in conditions]
    orig_cis = [summary[c]["orig_ci"] for c in conditions]
    sym_cis = [summary[c]["sym_ci"] for c in conditions]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    orig_err = [[a - ci[0] for a, ci in zip(orig_accs, orig_cis)],
                [ci[1] - a for a, ci in zip(orig_accs, orig_cis)]]
    sym_err = [[a - ci[0] for a, ci in zip(sym_accs, sym_cis)],
               [ci[1] - a for a, ci in zip(sym_accs, sym_cis)]]

    bars1 = ax.bar(x - width/2, [a*100 for a in orig_accs], width,
                   yerr=[[e*100 for e in orig_err[0]], [e*100 for e in orig_err[1]]],
                   label="GSM8K (Original)", color="#4C72B0", capsize=5)
    bars2 = ax.bar(x + width/2, [a*100 for a in sym_accs], width,
                   yerr=[[e*100 for e in sym_err[0]], [e*100 for e in sym_err[1]]],
                   label="GSM-Symbolic", color="#DD8452", capsize=5)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy on Standard vs Distribution-Shifted Problems")
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(c, c) for c in conditions])
    ax.legend()
    ax.set_ylim(0, 100)

    # Add value labels
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                       fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_comparison.png", dpi=150)
    plt.close()
    print(f"\nSaved: {PLOTS_DIR / 'accuracy_comparison.png'}")


def plot_robustness_ratio(summary):
    """Bar chart of robustness ratios."""
    conditions = list(summary.keys())
    labels = {"single": "Single Agent", "sc3": "SC@3", "debate": "Debate"}
    rrs = [summary[c]["robustness_ratio"] for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    bars = ax.bar([labels.get(c, c) for c in conditions],
                  [r * 100 for r in rrs], color=colors[:len(conditions)])

    ax.set_ylabel("Robustness Ratio (%)")
    ax.set_title("Robustness: GSM-Symbolic / GSM8K Accuracy")
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="Perfect robustness")
    ax.set_ylim(0, max(120, max(r*100 for r in rrs) + 10))
    ax.legend()

    for bar, rr in zip(bars, rrs):
        ax.annotate(f'{rr:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                   fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "robustness_ratio.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'robustness_ratio.png'}")


def plot_accuracy_drop(summary):
    """Accuracy drop from original to symbolic by condition."""
    conditions = list(summary.keys())
    labels = {"single": "Single Agent", "sc3": "SC@3", "debate": "Debate"}
    drops = [summary[c]["accuracy_drop"] * 100 for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    bars = ax.bar([labels.get(c, c) for c in conditions], drops, color=colors[:len(conditions)])

    ax.set_ylabel("Accuracy Drop (percentage points)")
    ax.set_title("Performance Degradation: GSM8K → GSM-Symbolic")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    for bar, drop in zip(bars, drops):
        ax.annotate(f'{drop:.1f}pp', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                   fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_drop.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'accuracy_drop.png'}")


def plot_debate_dynamics(dynamics):
    """Pie chart of debate outcome changes."""
    if not dynamics or dynamics["total"] == 0:
        return

    labels = ["Stayed Correct", "Flipped to Correct",
              "Flipped to Wrong", "Stayed Wrong"]
    sizes = [dynamics["stayed_correct"], dynamics["flip_to_correct"],
             dynamics["flip_to_wrong"], dynamics["stayed_wrong"]]
    colors = ["#55A868", "#7EC87E", "#E2A0A0", "#C44E52"]
    explode = (0, 0.1, 0.1, 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.0f%%', shadow=False, startangle=90,
        textprops={'fontsize': 11}
    )
    ax.set_title("Debate Dynamics: How Debate Changes Answers")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "debate_dynamics.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'debate_dynamics.png'}")


def plot_per_problem_comparison(all_results, summary):
    """Scatter plot: per-problem correctness across conditions."""
    if "single" not in all_results or "debate" not in all_results:
        return

    # Group by original_id
    single_by_id = {}
    debate_by_id = {}
    for r in all_results["single"]:
        oid = r["original_id"]
        if oid not in single_by_id:
            single_by_id[oid] = {"original": None, "symbolic": []}
        if r["type"] == "original":
            single_by_id[oid]["original"] = r["correct"]
        else:
            single_by_id[oid]["symbolic"].append(r["correct"])

    for r in all_results["debate"]:
        oid = r["original_id"]
        if oid not in debate_by_id:
            debate_by_id[oid] = {"original": None, "symbolic": []}
        if r["type"] == "original":
            debate_by_id[oid]["original"] = r["correct"]
        else:
            debate_by_id[oid]["symbolic"].append(r["correct"])

    # Per-problem symbolic accuracy
    common_ids = set(single_by_id.keys()) & set(debate_by_id.keys())
    single_sym_accs = []
    debate_sym_accs = []
    for oid in common_ids:
        s_sym = single_by_id[oid]["symbolic"]
        d_sym = debate_by_id[oid]["symbolic"]
        if s_sym and d_sym:
            single_sym_accs.append(np.mean(s_sym))
            debate_sym_accs.append(np.mean(d_sym))

    if not single_sym_accs:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(single_sym_accs, debate_sym_accs, alpha=0.6, s=80, color="#4C72B0")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label="y=x (no change)")
    ax.set_xlabel("Single Agent Accuracy (per problem)")
    ax.set_ylabel("Debate Accuracy (per problem)")
    ax.set_title("Per-Problem: Single Agent vs Debate on GSM-Symbolic")
    ax.legend()
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "per_problem_scatter.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'per_problem_scatter.png'}")


def generate_summary_table(summary, test_results):
    """Generate a markdown summary table."""
    labels = {"single": "Single Agent", "sc3": "SC@3", "debate": "Debate"}

    lines = [
        "## Results Summary Table\n",
        "| Condition | GSM8K Acc | GSM-Sym Acc | Robustness Ratio | Acc Drop |",
        "|-----------|-----------|-------------|------------------|----------|",
    ]
    for cond in summary:
        s = summary[cond]
        lines.append(
            f"| {labels.get(cond, cond)} | "
            f"{s['orig_acc']:.1%} [{s['orig_ci'][0]:.1%},{s['orig_ci'][1]:.1%}] | "
            f"{s['sym_acc']:.1%} [{s['sym_ci'][0]:.1%},{s['sym_ci'][1]:.1%}] | "
            f"{s['robustness_ratio']:.3f} | "
            f"{s['accuracy_drop']:.1%} |"
        )

    return "\n".join(lines)


def main():
    print("Loading results...")
    all_results = load_results()

    if not all_results:
        print("No results found. Run experiment.py first.")
        return

    print(f"Loaded results for conditions: {list(all_results.keys())}")

    # Robustness analysis
    summary = analyze_robustness(all_results)

    # Debate dynamics
    dynamics = analyze_debate_dynamics(all_results)

    # Statistical tests
    test_results = statistical_tests(all_results, summary)

    # Plots
    print("\nGenerating plots...")
    plot_accuracy_comparison(summary)
    plot_robustness_ratio(summary)
    plot_accuracy_drop(summary)
    plot_debate_dynamics(dynamics)
    plot_per_problem_comparison(all_results, summary)

    # Summary table
    table = generate_summary_table(summary, test_results)
    print("\n" + table)

    # Save analysis summary
    analysis = {
        "summary": {k: {sk: (sv if not isinstance(sv, tuple) else list(sv))
                        for sk, sv in v.items()} for k, v in summary.items()},
        "debate_dynamics": dynamics,
        "statistical_tests": {k: {sk: float(sv) if isinstance(sv, (np.floating, float)) else sv
                                   for sk, sv in v.items()} for k, v in test_results.items()},
    }
    with open(RESULTS_DIR / "analysis_summary.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nAnalysis saved to {RESULTS_DIR / 'analysis_summary.json'}")


if __name__ == "__main__":
    main()
