import torch

from ddgs import DDGS

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

from pathlib import Path
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

ADAPTER_PATH = str(Path(__file__).resolve().parent.parent / "qwen-fact-checker")


# ============================================================
# LOAD MODEL
# ============================================================

print("========================================")
print("LOADING FACT CHECKING MODEL")
print("========================================")

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True
)


tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("Loading Qwen...")

base_model = AutoModelForCausalLM.from_pretrained(

    BASE_MODEL,

    quantization_config=bnb_config,

    device_map="auto",

    dtype=torch.float16
)


print("Loading trained adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()

print("Model ready!")


# ============================================================
# WEB SEARCH
# ============================================================

def search_web(claim, max_results=5):

    print("\nSearching web...")
    print("Query:", claim)

    results = []

    try:

        with DDGS() as ddgs:

            search_results = ddgs.text(
                claim,
                max_results=max_results
            )

            for result in search_results:

                results.append({
                    "title": result.get("title", ""),
                    "body": result.get("body", ""),
                    "url": result.get("href", "")
                })

    except Exception as e:

        print("Search error:", e)

    return results


# ============================================================
# DISPLAY EVIDENCE
# ============================================================

def display_evidence(results):

    print("\n========================================")
    print("WEB EVIDENCE")
    print("========================================")

    if not results:

        print("No search results found.")

        return

    for i, result in enumerate(results, 1):

        print(f"\nSOURCE {i}")
        print("----------------------------------------")

        print("Title:")
        print(result["title"])

        print("\nEvidence:")
        print(result["body"])

        print("\nURL:")
        print(result["url"])


# ============================================================
# MODEL VERIFICATION
# ============================================================

def verify_claim(claim, evidence):

    evidence_text = ""

    for i, result in enumerate(evidence, 1):

        evidence_text += (
            f"\nSOURCE {i}\n"
            f"Title: {result['title']}\n"
            f"Evidence: {result['body']}\n"
        )


    messages = [

        {
            "role": "system",
            "content": (
                "You are an AI fact verification assistant. "
                "Use the provided web evidence to verify the claim. "
                "Classify the claim as exactly one of: "
                "SUPPORTED, REFUTED, PARTIALLY_SUPPORTED, or UNCERTAIN. "
                "Return the verdict first, followed by a short explanation."
            )
        },

        {
            "role": "user",
            "content": (
                f"CLAIM:\n"
                f"{claim}\n\n"
                f"WEB EVIDENCE:\n"
                f"{evidence_text}\n\n"
                f"Determine the correct verdict."
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
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)


    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=100,

            do_sample=False,

            pad_token_id=tokenizer.eos_token_id
        )


    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]


    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


    return response


# ============================================================
# COMPLETE FACT CHECK
# ============================================================

def fact_check(claim):

    print("\n========================================")
    print("FACT CHECK")
    print("========================================")

    print("\nCLAIM:")
    print(claim)


    # --------------------------------------------------------
    # Search web
    # --------------------------------------------------------

    evidence = search_web(claim)


    # --------------------------------------------------------
    # Display evidence
    # --------------------------------------------------------

    display_evidence(evidence)


    # --------------------------------------------------------
    # Verify using Qwen
    # --------------------------------------------------------

    if evidence:

        print("\n========================================")
        print("QWEN VERIFICATION")
        print("========================================")

        verdict = verify_claim(
            claim,
            evidence
        )

        print("\nMODEL RESULT:")
        print(verdict)

    else:

        print("\nNo evidence was found.")
        print("Unable to verify claim.")


# ============================================================
# INTERACTIVE MODE
# ============================================================

print("\n========================================")
print("WEB FACT CHECKER READY")
print("========================================")

print("Enter a factual claim.")
print("Type 'exit' to quit.")


while True:

    claim = input("\nClaim: ").strip()


    if claim.lower() == "exit":

        break


    if not claim:

        continue


    fact_check(claim)


print("\nFact checker stopped.")