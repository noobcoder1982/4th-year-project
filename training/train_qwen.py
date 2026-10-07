import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training
)

from trl import SFTTrainer


# ============================================================
# CONFIGURATION
# ============================================================

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TRAIN_FILE = str(SCRIPT_DIR / "train.jsonl")

OUTPUT_DIR = str(PROJECT_ROOT / "qwen-fact-checker")


# ============================================================
# GPU INFORMATION
# ============================================================

print("========================================")
print("GPU INFORMATION")
print("========================================")

print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )


# ============================================================
# LOAD DATASET
# ============================================================

print("\n========================================")
print("LOADING DATASET")
print("========================================")

dataset = load_dataset(
    "json",
    data_files=TRAIN_FILE,
    split="train"
)

print(
    "Training examples:",
    len(dataset)
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\n========================================")
print("LOADING TOKENIZER")
print("========================================")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded!")


# ============================================================
# 4-BIT QUANTIZATION
# ============================================================

print("\n========================================")
print("CONFIGURING QLoRA")
print("========================================")

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True
)

print("4-bit NF4 configured!")


# ============================================================
# LOAD QWEN MODEL
# ============================================================

print("\n========================================")
print("LOADING QWEN")
print("========================================")

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto",

    dtype=torch.float16
)

print("Model loaded!")


# ============================================================
# PREPARE MODEL FOR QLoRA
# ============================================================

print("\n========================================")
print("PREPARING MODEL FOR QLoRA")
print("========================================")

model = prepare_model_for_kbit_training(
    model
)

model.config.use_cache = False

print("Model prepared!")


# ============================================================
# LoRA CONFIGURATION
# ============================================================

print("\n========================================")
print("CONFIGURING LoRA")
print("========================================")

lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)

print("LoRA configured!")


# ============================================================
# FORMAT DATASET
# ============================================================

def formatting_func(example):

    messages = example["messages"]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

print("\n========================================")
print("TRAINING CONFIGURATION")
print("========================================")

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=5,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    # ========================================================
    # IMPORTANT:
    # Disable AMP completely.
    #
    # The current PyTorch/Accelerate combination is triggering
    # a CUDA BF16 gradient-scaler error.
    # ========================================================

    fp16=False,

    bf16=False,

    # ========================================================
    # Gradient clipping disabled.
    # ========================================================

    max_grad_norm=0.0,

    logging_steps=1,

    save_strategy="epoch",

    optim="paged_adamw_8bit",

    report_to="none",

    gradient_checkpointing=True
)

print("Training arguments configured!")

print("Mixed precision: DISABLED")
print("FP16 AMP: DISABLED")
print("BF16 AMP: DISABLED")
print("Gradient clipping: DISABLED")


# ============================================================
# CREATE TRAINER
# ============================================================

print("\n========================================")
print("CREATING TRAINER")
print("========================================")

trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=dataset,

    peft_config=lora_config,

    formatting_func=formatting_func
)

print("Trainer created!")


# ============================================================
# START TRAINING
# ============================================================

print("\n========================================")
print("STARTING QLoRA TRAINING")
print("========================================")

trainer.train()


# ============================================================
# SAVE MODEL
# ============================================================

print("\n========================================")
print("SAVING MODEL")
print("========================================")

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(
    f"Model saved to: {OUTPUT_DIR}"
)