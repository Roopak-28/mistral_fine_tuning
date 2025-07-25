import argparse
import json
from transformers import AutoTokenizer
from datasets import Dataset
import os

parser = argparse.ArgumentParser()
parser.add_argument("--jsonl_path", type=str, required=True)
parser.add_argument("--tokenizer_name", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--max_length", type=int, default=512)
args = parser.parse_args()

# Load JSONL
with open(args.jsonl_path, encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]

dataset = Dataset.from_list(lines)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=args.max_length, padding="max_length")

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
os.makedirs(args.output_path, exist_ok=True)
tokenized_dataset.save_to_disk(args.output_path)

print(f"Tokenized dataset saved to {args.output_path}")
