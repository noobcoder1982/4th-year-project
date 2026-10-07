import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


# ============================================================
# GPU Check
# ============================================================

print("========================================")
print("GPU INFORMATION")
print("========================================")

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2
        ),
        "GB"
    )


# ============================================================
# Load Tokenizer
# ============================================================

print("\n========================================")
print("LOADING TOKENIZER")
print("========================================")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded!")


# ============================================================
# 4-bit Quantization
# ============================================================

print("\n========================================")
print("CONFIGURING 4-BIT QUANTIZATION")
print("========================================")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("4-bit NF4 configured!")


# ============================================================
# Load Model
# ============================================================

print("\n========================================")
print("LOADING QWEN 2.5 3B")
print("========================================")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto"
)

print("\nModel loaded successfully!")


# ============================================================
# Test Inference
# ============================================================

print("\n========================================")
print("TESTING MODEL")
print("========================================")

messages = [
    {
        "role": "system",
        "content": (
            "You are a fact verification assistant. "
            "Determine whether a factual statement is "
            "supported, refuted, or partially supported."
        )
    },
    {
        "role": "user",
        "content": (
            "Verify this statement:\n\n"
            "The Earth is the third planet from the Sun."
        )
    }
]


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)


with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.1
    )


result = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)


print("\nModel response:")
print(result)

print("\n========================================")
print("QWEN TEST COMPLETE")
print("========================================")