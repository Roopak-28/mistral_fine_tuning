import os
import json
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# ========= PARAMETERS =========
# You can also set these with os.environ[...] if running in a Kubeflow container
dataset_name = "squad"
split = "train"
max_samples = 1000
csv_path = "/data/qna.csv"
output_jsonl = "/data/mistral.jsonl"
question_column = "question"
answer_column = "answer"
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
output_dir = "/data/peft_lora_model"

# ========= STEP 1: Download and Save QnA as CSV =========
print("Loading HuggingFace QnA dataset...")
ds = load_dataset(dataset_name, split=split)
df = ds.to_pandas()
if max_samples and max_samples < len(df):
    df = df.sample(n=max_samples, random_state=42)
df.to_csv(csv_path, index=False)
print(f"Saved {len(df)} rows to {csv_path}")

# ========= STEP 2: Convert to Mistral JSONL =========
print("Converting CSV to Mistral JSONL...")
with open(output_jsonl, "w") as fout:
    for _, row in df.iterrows():
        if pd.isna(row[question_column]) or pd.isna(row[answer_column]):
            continue  # skip blanks
        record = {
            "messages": [
                {"role": "user", "content": str(row[question_column])},
                {"role": "assistant", "content": str(row[answer_column])}
            ]
        }
        fout.write(json.dumps(record) + "\n")
print(f"Wrote {df.shape[0]} lines to {output_jsonl}")

# ========= STEP 3: Train Mistral LLM with LoRA/PEFT =========
print("Starting LoRA/PEFT fine-tuning...")

def jsonl_gen(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)

dataset = Dataset.from_generator(lambda: jsonl_gen(output_jsonl))

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

def preprocess(example):
    input_text = example["messages"][0]["content"]
    target_text = example["messages"][1]["content"]
    return tokenizer(input_text, text_target=target_text, truncation=True, padding='max_length', max_length=256)

print("Tokenizing...")
tokenized = dataset.map(preprocess, remove_columns=dataset.column_names)

args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_steps=10,
    logging_steps=10,
    fp16=True,
    overwrite_output_dir=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized
)
trainer.train()
model.save_pretrained(output_dir)
print(f"LoRA adapter/model saved to {output_dir}")

print("Done!")
