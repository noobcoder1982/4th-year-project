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

from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

ADAPTER_PATH = str(Path(_root) / "qwen-fact-checker")


# ============================================================
# GPU INFORMATION
# ============================================================

print("========================================")
print("LOADING TRAINED FACT CHECKER")
print("========================================")

print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 4-BIT CONFIGURATION
# ============================================================

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True
)


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("Loading base Qwen model...")

base_model = AutoModelForCausalLM.from_pretrained(

    BASE_MODEL,

    quantization_config=bnb_config,

    device_map="auto",

    dtype=torch.float16
)


# ============================================================
# LOAD TRAINED LoRA ADAPTER
# ============================================================

print("Loading trained LoRA adapter...")

model = PeftModel.from_pretrained(

    base_model,

    ADAPTER_PATH
)

model.eval()

print("\n========================================")
print("TRAINED MODEL LOADED!")
print("========================================")


# ============================================================
# FACT CHECK FUNCTION
# ============================================================

def fact_check(statement):

    messages = [

        {
            "role": "system",
            "content": (
                "You are a fact verification assistant. "
                "Determine whether a factual statement is "
                "SUPPORTED, REFUTED, or PARTIALLY_SUPPORTED. "
                "Return only the verdict."
            )
        },

        {
            "role": "user",
            "content": (
                f"Verify this statement:\n\n{statement}"
            )
        }

    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=20,

            do_sample=False,

            temperature=None,

            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return response


# ============================================================
# TEST CLAIMS
# ============================================================

test_claims = [

    "The Earth is the third planet from the Sun.",

    "Cleopatra was the Queen of Rome.",

    "Water freezes at 0 degrees Celsius under normal atmospheric pressure.",

    "The Moon is made entirely of cheese."

]


# ============================================================
# RUN TESTS
# ============================================================

print("\n========================================")
print("RUNNING FACT-CHECK TESTS")
print("========================================")

for i, claim in enumerate(test_claims, 1):

    print("\n----------------------------------------")

    print(f"TEST {i}")

    print("----------------------------------------")

    print("Claim:")
    print(claim)

    print("\nModel verdict:")

    result = fact_check(claim)

    print(result)


# ============================================================
# INTERACTIVE MODE
# ============================================================

print("\n========================================")
print("INTERACTIVE FACT CHECKER")
print("========================================")

print("Type a claim to test the model.")
print("Type 'exit' to quit.")

while True:

    claim = input("\nClaim: ").strip()

    if claim.lower() == "exit":
        break

    if not claim:
        continue

    result = fact_check(claim)

    print("\nVerdict:")
    print(result)


print("\nTesting finished.")