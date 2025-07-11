# qna_to_mistral_format.py

import os
from datasets import load_dataset
import json

# === CONFIGURABLE PARAMETERS ===
dataset_name = "squad"           # Any HuggingFace QnA dataset, e.g., 'squad', 'quora', etc.
max_samples = 2000               # Number of samples to process (set None for all)
output_path = "qna_mistral.jsonl"  # Output file in JSONL (one record per line)

# === 1. Load QnA Dataset ===
dataset = load_dataset(dataset_name, split=f"train[:{max_samples}]" if max_samples else "train")

# === 2. Convert to Mistral "Instruction" Format ===
def format_example(example):
    # Picks the first answer for simplicity
    return {
        "text": f"<s>[INST] {example['question']} [/INST] {example['answers']['text'][0]}</s>"
    }

formatted_data = [format_example(x) for x in dataset]

# === 3. Save as JSONL ===
with open(output_path, "w", encoding="utf-8") as f:
    for record in formatted_data:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved {len(formatted_data)} samples in Mistral instruction format to {output_path}")
