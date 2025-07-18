name: Tokenize Mistral JSONL Dataset
description: Loads a Mistral-format QnA JSONL, tokenizes it using a HuggingFace tokenizer, and saves it as a HuggingFace dataset.

inputs:
  - { name: jsonl_path, type: String, description: "Path to Mistral-format JSONL file" }
  - { name: tokenizer_name, type: String, description: "HuggingFace tokenizer name (e.g. mistralai/Mistral-7B-Instruct-v0.2)" }
  - { name: max_length, type: Integer, description: "Maximum token length", default: 512 }

outputs:
  - { name: tokenized_path, type: Path, description: "Path to saved HuggingFace dataset directory" }

implementation:
  container:
    image: python:3.10
    command:
      - python3
      - -u
      - -c
      - |
        import argparse
        import json
        import os
        from datasets import Dataset
        from transformers import AutoTokenizer

        parser = argparse.ArgumentParser()
        parser.add_argument("--jsonl_path", type=str, required=True)
        parser.add_argument("--tokenizer_name", type=str, required=True)
        parser.add_argument("--max_length", type=int, required=True)
        parser.add_argument("--tokenized_path", type=str, required=True)
        args = parser.parse_args()

        with open(args.jsonl_path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        dataset = Dataset.from_list(lines)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

        def tokenize_fn(batch):
            return tokenizer(
                batch["text"], 
                truncation=True, 
                max_length=args.max_length, 
                padding="max_length"
            )

        tokenized_dataset = dataset.map(tokenize_fn, batched=True)
        os.makedirs(args.tokenized_path, exist_ok=True)
        tokenized_dataset.save_to_disk(args.tokenized_path)
        print(f"Tokenized dataset saved to {args.tokenized_path}")
    args:
      - --jsonl_path
      - { inputValue: jsonl_path }
      - --tokenizer_name
      - { inputValue: tokenizer_name }
      - --max_length
      - { inputValue: max_length }
      - --tokenized_path
      - { outputPath: tokenized_path }
