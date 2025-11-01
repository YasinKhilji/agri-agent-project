import torch
import gc
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import os

print("--- Starting local fine-tuning script ---")

# --- 1. Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_FILE = "finetune_data.jsonl"
ADAPTER_SAVE_PATH = "./model/planner_lora_adapter"

# --- 2. Load Dataset ---
print(f"Loading dataset from {DATASET_FILE}...")
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

# --- 3. Format Dataset with Chat Template ---
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token 

def format_prompt(example):
    prompt = example['prompt']
    completion = example['completion']
    return {
        "text": f"<|system|>\nYou are an AI assistant that only replies with JSON.<|user|>\n{prompt}<|assistant|>\n{completion}{tokenizer.eos_token}"
    }

print("Formatting dataset...")
formatted_dataset = dataset.map(format_prompt)

# --- 4. Load Model (TinyLlama) ---
print(f"Loading base model from {MODEL_NAME}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print("Base model loaded to GPU.")

# --- 5. LoRA Configuration (Optimized) ---
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
print("PEFT config created.")

# --- 6. Training Arguments (Optimized for 16GB) ---
training_args = TrainingArguments(
    output_dir="./results-planner",
    num_train_epochs=15, 
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=True,
    report_to="none",
)
print("Training arguments set.")

# --- 7. Initialize Trainer (Correct for trl 0.8.6) ---
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    peft_config=peft_config,
    dataset_text_field="text",  # <-- THIS IS NOW CORRECT
    max_seq_length=256,        # <-- THIS IS NOW CORRECT
    tokenizer=tokenizer,
    args=training_args,
)
print("Trainer initialized.")

# --- 8. Start Training ---
print("Cleaning CUDA cache...")
torch.cuda.empty_cache()
gc.collect()

print("--- Starting training ---")
trainer.train()
print("--- Training complete ---")

# --- 9. Save the LoRA Adapter ---
os.makedirs(ADAPTER_SAVE_PATH, exist_ok=True)
print(f"Saving LoRA adapter to {ADAPTER_SAVE_PATH}...")
trainer.model.save_pretrained(ADAPTER_SAVE_PATH)
tokenizer.save_pretrained(ADAPTER_SAVE_PATH)

print("--- Fine-tuning script finished successfully! ---")