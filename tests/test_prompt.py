import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained("./qwen-fact-checker")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16
)
model = PeftModel.from_pretrained(base_model, "./qwen-fact-checker")
model.eval()

claim = "The Sun is a star located at the center of our solar system."
evidence_text = """
SOURCE 1 (britannica.com)
TITLE: Sun | Definition, Facts, & Temperature
EVIDENCE: The Sun is the star at the center of the solar system. Earth and other components of the solar system revolve around it.

SOURCE 2 (science.nasa.gov)
TITLE: Our Sun: Facts - NASA Science
EVIDENCE: The Sun is the only star in our solar system. Its gravity holds the solar system together, keeping everything in orbit around it.
"""

system_prompt = (
    "You are an expert fact-verification judge.\n"
    "Evaluate whether the CLAIM is substantiated by the WEB EVIDENCE.\n\n"
    "VERDICT LABELS:\n"
    "- SUPPORTED: The evidence substantiates the factual substance of the claim. "
    "Semantically equivalent phrasing (e.g., 'star at the center of the solar system' vs. 'star located at the center of our solar system') "
    "counts as FULL SUPPORT even if the exact wording differs.\n"
    "- REFUTED: The evidence directly contradicts or disproves the claim.\n"
    "- PARTIALLY_SUPPORTED: The claim combines multiple distinct factual assertions, and the evidence verifies one assertion but contradicts another.\n"
    "- UNCERTAIN: The evidence is insufficient, conflicting, or lacks enough information.\n\n"
    "OUTPUT FORMAT:\n"
    "VERDICT: <SUPPORTED | REFUTED | PARTIALLY_SUPPORTED | UNCERTAIN>\n"
    "EXPLANATION: <1-2 concise sentences explaining the verdict and explicitly citing the evidence sources (e.g. NASA, Britannica).>"
)

user_prompt = f"CLAIM:\n{claim}\n\nWEB EVIDENCE:\n{evidence_text}\n\nVERDICT AND EXPLANATION:"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
res = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
print("OUTPUT:")
print(res)
