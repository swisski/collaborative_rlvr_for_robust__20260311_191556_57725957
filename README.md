# Collaborative RLVR for Robust Reasoning

Testing whether multi-agent debate improves robustness of LLM mathematical reasoning under distribution shifts (GSM8K vs GSM-Symbolic).

## Key Findings

- **Inference-time debate does not improve robustness** for small (1.5B) math models -- accuracy and robustness ratios were statistically equivalent across single-agent, self-consistency, and debate conditions
- **Surprisingly, the small model showed no robustness degradation** on its own (60% on both GSM8K and GSM-Symbolic), unlike larger models which show up to 65% drops
- **Debate was net-neutral**: corrected 2 errors but introduced 4 new ones, with 64% of problems showing initial agent agreement (limiting debate utility)
- **Problem difficulty dominates method choice**: 67% of problems were always correct or always wrong regardless of inference strategy
- **Training-time collaborative RLVR remains the promising direction**, consistent with SDRL (Liu et al., 2026) finding that training for debate is essential

## Results Summary

| Condition | GSM8K Acc | GSM-Sym Acc | Robustness Ratio |
|-----------|-----------|-------------|------------------|
| Single Agent | 60.0% | 60.0% | 1.000 |
| SC@3 | 53.3% | 56.7% | 1.062 |
| Debate (1 round) | 53.3% | 56.7% | 1.062 |

## Reproduce

```bash
# Setup
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers accelerate datasets scipy matplotlib numpy

# Run experiments (~2 hours on 32-core CPU)
OMP_NUM_THREADS=32 python src/experiment.py

# Analyze results
python src/analyze_results.py
```

## File Structure

```
.
├── REPORT.md                    # Full research report with results
├── README.md                    # This file
├── planning.md                  # Research plan and experimental design
├── literature_review.md         # Comprehensive literature review
├── resources.md                 # Catalog of papers, datasets, code
├── src/
│   ├── experiment.py            # Main experiment (single-agent, SC, debate)
│   ├── analyze_results.py       # Statistical analysis and visualization
│   └── benchmark_speed.py       # CPU inference speed benchmark
├── results/
│   ├── config.json              # Experiment configuration
│   ├── problems.json            # Selected problems
│   ├── single_agent_results.json
│   ├── sc_results.json
│   ├── debate_results.json
│   ├── analysis_summary.json
│   └── plots/                   # Visualizations (PNG)
├── datasets/                    # Pre-downloaded datasets (Arrow format)
│   ├── gsm8k/
│   ├── gsm_symbolic/
│   ├── math/
│   ├── math_500/
│   └── dapo_math_17k/
├── papers/                      # 33 research papers (PDF)
└── code/                        # 6 code repositories
```

## Model & Setup

- **Model:** Qwen2.5-Math-1.5B-Instruct with dynamic INT8 quantization
- **Hardware:** 32-core CPU, 46GB RAM (no GPU)
- **Inference speed:** ~15 tok/s (4.4x speedup from quantization)
- **Datasets:** GSM8K (15 problems) + GSM-Symbolic (30 symbolic variants)
- **Total runtime:** 128 minutes

See [REPORT.md](REPORT.md) for the full research report with detailed analysis.
