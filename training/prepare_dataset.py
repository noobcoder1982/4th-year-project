import pandas as pd
import json
from huggingface_hub import hf_hub_download


# ============================================================
# 1. Download / locate dataset
# ============================================================

print("Getting dataset...")

file_path = hf_hub_download(
    repo_id="Yogeshwaran10/fact-verification-dataset",
    filename="data/train-00000-of-00001.parquet",
    repo_type="dataset"
)


# ============================================================
# 2. Load dataset
# ============================================================

print("\nReading dataset...")

df = pd.read_parquet(file_path)

print(f"Loaded {len(df)} examples")


# ============================================================
# 3. Show label distribution
# ============================================================

print("\nLabel distribution:")
print(df["label"].value_counts())


# ============================================================
# 4. Shuffle dataset
# ============================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 5. Train / Test split
# ============================================================

split = int(len(df) * 0.8)

train_df = df.iloc[:split]
test_df = df.iloc[split:]


# ============================================================
# 6. Convert labels and create Qwen format
# ============================================================

def convert_row(row):

    statement = row["statement"]
    label = row["label"]

    label_upper = label.upper()

    if label_upper == "CORRECT":
        verdict = "SUPPORTED"

    elif label_upper == "INCORRECT":
        verdict = "REFUTED"

    elif label_upper == "PARTIALLY CORRECT":
        verdict = "PARTIALLY_SUPPORTED"

    else:
        verdict = "UNCERTAIN"

    return {
        "messages": [
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
                    f"Verify this statement:\n\n{statement}"
                )
            },
            {
                "role": "assistant",
                "content": verdict
            }
        ]
    }


from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent

# ============================================================
# 7. Create training JSONL
# ============================================================

train_file = str(SCRIPT_DIR / "train.jsonl")

with open(train_file, "w", encoding="utf-8") as f:

    for _, row in train_df.iterrows():

        example = convert_row(row)

        f.write(
            json.dumps(
                example,
                ensure_ascii=False
            ) + "\n"
        )


# ============================================================
# 8. Create testing JSONL
# ============================================================

test_file = str(SCRIPT_DIR / "test.jsonl")

with open(test_file, "w", encoding="utf-8") as f:

    for _, row in test_df.iterrows():

        example = convert_row(row)

        f.write(
            json.dumps(
                example,
                ensure_ascii=False
            ) + "\n"
        )


# ============================================================
# 9. Final information
# ============================================================

print("\n========================================")
print("DATASET PREPARATION COMPLETE")
print("========================================")

print(f"\nTraining examples: {len(train_df)}")
print(f"Testing examples:  {len(test_df)}")

print("\nCreated files:")
print(f"  {train_file}")
print(f"  {test_file}")


# ============================================================
# 10. Display example
# ============================================================

print("\nExample training record:")

print(
    json.dumps(
        convert_row(train_df.iloc[0]),
        indent=2,
        ensure_ascii=False
    )
)