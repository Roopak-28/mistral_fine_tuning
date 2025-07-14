import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_name", type=str, required=True, help="HF model ID or path to base model (e.g. mistralai/Mistral-7B-Instruct-v0.2)")
parser.add_argument("--peft_model_dir", type=str, required=True, help="Directory of the trained LoRA adapter")
parser.add_argument("--merged_model_dir", type=str, required=True, help="Where to save the merged model")
args = parser.parse_args()

print("Loading base model:", args.base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
print("Loading PEFT adapter from:", args.peft_model_dir)
peft_model = PeftModel.from_pretrained(base_model, args.peft_model_dir)

print("Merging LoRA adapter into base model...")
merged_model = peft_model.merge_and_unload()
os.makedirs(args.merged_model_dir, exist_ok=True)
merged_model.save_pretrained(args.merged_model_dir)
tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True)
tokenizer.save_pretrained(args.merged_model_dir)
print(f"Merged model saved to {args.merged_model_dir}")
