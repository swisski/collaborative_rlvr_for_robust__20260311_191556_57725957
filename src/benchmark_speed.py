"""Benchmark inference speed to calibrate experiment size."""
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"

print(f"Loading model {MODEL_NAME}...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)
model.eval()
print(f"Model loaded in {time.time()-t0:.1f}s")

# Test problem
problem = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

messages = [
    {"role": "system", "content": "Solve the following math problem step by step. Put your final answer after 'Answer:'."},
    {"role": "user", "content": problem}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

# Benchmark single generation
print("\nBenchmarking single generation (max_new_tokens=256)...")
t0 = time.time()
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        temperature=1.0,
    )
t1 = time.time()
gen_tokens = out.shape[1] - inputs['input_ids'].shape[1]
print(f"Generated {gen_tokens} tokens in {t1-t0:.1f}s ({gen_tokens/(t1-t0):.1f} tok/s)")
response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"Response: {response[:200]}...")

# Benchmark with sampling
print("\nBenchmarking sampled generation (temp=0.7)...")
t0 = time.time()
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
t1 = time.time()
gen_tokens = out.shape[1] - inputs['input_ids'].shape[1]
print(f"Generated {gen_tokens} tokens in {t1-t0:.1f}s ({gen_tokens/(t1-t0):.1f} tok/s)")

print("\nBenchmark complete. Use these numbers to estimate total experiment time.")
