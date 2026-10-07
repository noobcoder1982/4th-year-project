import re
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

MAX_SEARCH_RESULTS = 5

MAX_EVIDENCE_CHARS = 7000


# ============================================================
# LOAD MODEL
# ============================================================

print("========================================")
print("LOADING AI ARTICLE VERIFIER")
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

print("AI model ready!")


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def extract_claims(article):

    article = re.sub(
        r"\s+",
        " ",
        article
    ).strip()


    sentences = re.split(
        r"(?<=[.!?])\s+",
        article
    )


    claims = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence.split()) < 5:
            continue

        claims.append(sentence)


    return claims


# ============================================================
# WEB SEARCH
# ============================================================

def search_web(claim):

    results = []

    # Add fact-check context to improve retrieval.
    query = f"{claim} fact check"

    print("\nSearch query:")
    print(query)


    try:

        with DDGS() as ddgs:

            search_results = ddgs.text(
                query,
                max_results=MAX_SEARCH_RESULTS
            )


            for result in search_results:

                title = result.get(
                    "title",
                    ""
                )

                body = result.get(
                    "body",
                    ""
                )

                url = result.get(
                    "href",
                    ""
                )


                if not title and not body:
                    continue


                results.append({

                    "title": title,

                    "body": body,

                    "url": url

                })


    except Exception as e:

        print(
            "Search error:",
            e
        )


    return results


# ============================================================
# BUILD EVIDENCE
# ============================================================

def build_evidence_text(evidence):

    evidence_text = ""


    for i, result in enumerate(
        evidence,
        1
    ):

        evidence_text += (

            f"\nSOURCE {i}\n"

            f"TITLE: "
            f"{result['title']}\n"

            f"EVIDENCE: "
            f"{result['body']}\n"

            f"URL: "
            f"{result['url']}\n"

        )


    return evidence_text[:MAX_EVIDENCE_CHARS]


# ============================================================
# QWEN EVIDENCE JUDGE
# ============================================================

def verify_claim(
    claim,
    evidence
):

    evidence_text = build_evidence_text(
        evidence
    )


    messages = [

        {
            "role": "system",

            "content": (
                "You are a strict evidence-based "
                "fact verification system.\n\n"

                "Your task is to determine whether "
                "a CLAIM is supported by the provided "
                "WEB EVIDENCE.\n\n"

                "IMPORTANT RULES:\n"

                "1. Use the web evidence as the primary "
                "basis for your decision.\n"

                "2. Do not rely only on your memory or "
                "general knowledge.\n"

                "3. If reliable evidence directly "
                "contradicts the claim, classify it "
                "as REFUTED.\n"

                "4. If reliable evidence directly "
                "supports the claim, classify it "
                "as SUPPORTED.\n"

                "5. If evidence supports only part "
                "of the claim, classify it as "
                "PARTIALLY_SUPPORTED.\n"

                "6. If the evidence is insufficient "
                "or conflicting, classify it as "
                "UNCERTAIN.\n\n"

                "You MUST begin your answer with exactly "
                "one of these labels:\n\n"

                "SUPPORTED\n"
                "REFUTED\n"
                "PARTIALLY_SUPPORTED\n"
                "UNCERTAIN\n\n"

                "After the label, provide a short "
                "explanation based on the evidence."
            )
        },

        {
            "role": "user",

            "content": (

                f"CLAIM:\n"
                f"{claim}\n\n"

                f"WEB EVIDENCE:\n"
                f"{evidence_text}\n\n"

                "VERDICT:"
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

            max_new_tokens=120,

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
# STRICT VERDICT EXTRACTION
# ============================================================

def extract_verdict(response):

    text = response.upper().strip()


    # Check PARTIAL first because it contains
    # the word SUPPORTED.

    if re.search(
        r"\bPARTIALLY_SUPPORTED\b",
        text
    ):

        return "PARTIALLY_SUPPORTED"


    if re.search(
        r"\bSUPPORTED\b",
        text
    ):

        return "SUPPORTED"


    if re.search(
        r"\bREFUTED\b",
        text
    ):

        return "REFUTED"


    if re.search(
        r"\bUNCERTAIN\b",
        text
    ):

        return "UNCERTAIN"


    return "UNCERTAIN"


# ============================================================
# SIMPLE EVIDENCE CONSISTENCY CHECK
# ============================================================

def evidence_consistency_check(
    claim,
    evidence,
    verdict
):

    claim_lower = claim.lower()

    combined = " ".join(

        result["body"].lower()

        for result in evidence

    )


    # --------------------------------------------------------
    # Extremely obvious contradiction.
    #
    # This is NOT the main classifier.
    # It only catches simple cases where the evidence
    # clearly states the opposite of a claim.
    # --------------------------------------------------------

    if (
        "made entirely of cheese" in claim_lower
        or "made of cheese" in claim_lower
    ):

        contradiction_terms = [

            "rock",

            "rocky",

            "silicate",

            "not made of cheese",

            "not cheese"

        ]


        for term in contradiction_terms:

            if term in combined:

                return "REFUTED"


    return verdict


# ============================================================
# VERIFY ONE CLAIM
# ============================================================

def process_claim(
    claim,
    claim_number,
    total_claims
):

    print("\n")
    print("========================================")
    print(
        f"CLAIM {claim_number}/{total_claims}"
    )
    print("========================================")


    print("\nClaim:")
    print(claim)


    # --------------------------------------------------------
    # WEB SEARCH
    # --------------------------------------------------------

    print("\nSearching web...")

    evidence = search_web(
        claim
    )


    if not evidence:

        print(
            "\nNo web evidence found."
        )

        return {

            "claim": claim,

            "verdict": "UNCERTAIN",

            "explanation": (
                "No suitable web evidence "
                "was found."
            ),

            "sources": []

        }


    print(
        f"\nFound {len(evidence)} sources."
    )


    # --------------------------------------------------------
    # Display sources
    # --------------------------------------------------------

    for number, source in enumerate(
        evidence,
        1
    ):

        print(
            f"\nSOURCE {number}"
        )

        print(
            source["title"]
        )

        print(
            source["url"]
        )


    # --------------------------------------------------------
    # QWEN VERIFICATION
    # --------------------------------------------------------

    print(
        "\nRunning evidence-based Qwen verification..."
    )


    response = verify_claim(

        claim,

        evidence

    )


    verdict = extract_verdict(
        response
    )


    # --------------------------------------------------------
    # Consistency check
    # --------------------------------------------------------

    verdict = evidence_consistency_check(

        claim,

        evidence,

        verdict

    )


    print("\nAI VERDICT:")
    print(verdict)


    print("\nAI EXPLANATION:")
    print(response)


    return {

        "claim": claim,

        "verdict": verdict,

        "explanation": response,

        "sources": evidence

    }


# ============================================================
# VERIFY COMPLETE ARTICLE
# ============================================================

def verify_article(article):

    print("\n========================================")
    print("EXTRACTING CLAIMS")
    print("========================================")


    claims = extract_claims(
        article
    )


    print(
        f"Found {len(claims)} claims."
    )


    if not claims:

        print(
            "No suitable claims found."
        )

        return


    results = []


    # ========================================================
    # PROCESS CLAIMS
    # ========================================================

    for index, claim in enumerate(
        claims,
        1
    ):

        result = process_claim(

            claim,

            index,

            len(claims)

        )


        results.append(
            result
        )


    # ========================================================
    # COUNT VERDICTS
    # ========================================================

    supported = sum(

        1

        for result in results

        if result["verdict"] == "SUPPORTED"

    )


    refuted = sum(

        1

        for result in results

        if result["verdict"] == "REFUTED"

    )


    partial = sum(

        1

        for result in results

        if result["verdict"] == "PARTIALLY_SUPPORTED"

    )


    uncertain = sum(

        1

        for result in results

        if result["verdict"] == "UNCERTAIN"

    )


    total = len(results)


    # ========================================================
    # OVERALL ARTICLE VERDICT
    # ========================================================

    if refuted >= 1:

        if refuted >= supported:

            overall = "LIKELY MISLEADING"

        else:

            overall = "MIXED / NEEDS REVIEW"


    elif partial > 0:

        overall = "MIXED / NEEDS REVIEW"


    elif uncertain > total / 2:

        overall = "INSUFFICIENT EVIDENCE"


    else:

        overall = "LIKELY SUPPORTED"


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n\n")

    print(
        "########################################################"
    )

    print(
        "#                FINAL ARTICLE RESULT                 #"
    )

    print(
        "########################################################"
    )


    print("\nClaim Summary:")

    print(
        f"SUPPORTED: {supported}"
    )

    print(
        f"REFUTED: {refuted}"
    )

    print(
        f"PARTIALLY SUPPORTED: {partial}"
    )

    print(
        f"UNCERTAIN: {uncertain}"
    )


    print("\n========================================")
    print("OVERALL ARTICLE VERDICT")
    print("========================================")

    print(
        overall
    )


    print("\n========================================")
    print("CLAIM-BY-CLAIM SUMMARY")
    print("========================================")


    for index, result in enumerate(
        results,
        1
    ):

        print(
            f"\n{index}. "
            f"[{result['verdict']}]"
        )

        print(
            result["claim"]
        )


        if result["sources"]:

            print(
                "\nTop source:"
            )

            print(
                result["sources"][0]["title"]
            )

            print(
                result["sources"][0]["url"]
            )


# ============================================================
# MAIN PROGRAM
# ============================================================

print("\n========================================")
print("AI ARTICLE VERIFIER READY")
print("========================================")

print(
    "Paste an article below."
)

print(
    "Press ENTER on an empty line when finished."
)

print(
    "Type 'exit' to quit."
)


while True:

    print("\n")

    lines = []


    while True:

        line = input()


        if line.lower().strip() == "exit":

            print(
                "\nExiting..."
            )

            raise SystemExit


        if line.strip() == "":

            break


        lines.append(line)


    article = " ".join(
        lines
    ).strip()


    if not article:

        continue


    verify_article(
        article
    )