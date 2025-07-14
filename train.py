import argparse
import json
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--jsonl_path', type=str, required=True)
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--lora_r', type=int, default=8)
parser.add_argument('--lora_alpha', type=int, default=32)
parser.add_argument('--lora_dropout', type=float, default=0.05)
args = parser.parse_args()

print("Loading data...")
with open(args.jsonl_path, encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]
dataset = Dataset.from_list(lines)

tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=args.max_length, padding="max_length")

print("Tokenizing...")
tokenized_ds = dataset.map(tokenize_fn, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    report_to="none",
    fp16=True if torch.cuda.is_available() else False,
)

print("Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=data_collator,
)
trainer.train()
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print("PEFT LoRA model saved to", args.output_dir)

