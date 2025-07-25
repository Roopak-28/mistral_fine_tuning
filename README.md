
# Mistral Fine-Tuning Pipeline (Kubeflow/Elyra)

This repository contains a Kubeflow/Elyra pipeline for **fine-tuning a Mistral LLM using PEFT (LoRA)** adapters, following a modular, reusable, and open-source stack.

---

## **Pipeline Steps Overview**

1. **Load HuggingFace Dataset**
   Loads any HuggingFace QnA dataset and saves it as a CSV for downstream processing.

2. **Convert CSV to Mistral JSONL**
   Converts the QnA CSV file (with columns like `question` and `answer`) to a Mistral instruction-tuning JSONL format.

3. **Train the Mistral Model with PEFT (LoRA)**
   Fine-tunes a Mistral LLM using the PEFT LoRA method, saving only the adapter weights.

4. **Merge LoRA Adapter into Base Model**
   Merges the trained LoRA/PEFT adapter back into the base Mistral model, saving a full, standalone HuggingFace model directory.

---

## **How to Run the Pipeline**

### **1. Loading the Dataset**

* Component: `Load HuggingFace Dataset to CSV`
* **Inputs:**

  * `dataset_name`: e.g. `squad`
  * `split`: e.g. `train`
  * `max_samples`: e.g. `2000` (or `0` for all)
* **Output:**

  * `csv_path`: Path to the saved CSV (e.g. `/data/qna.csv`)

---

### **2. Converting CSV to Mistral Format**

* Component: `Convert CSV to Mistral JSONL`
* **Inputs:**

  * `csv_path`: Path to input CSV (from previous step)
  * `output_jsonl`: Path for JSONL output (e.g. `/data/mistral.jsonl`)
  * `question_column`: Column name for questions (default: `question`)
  * `answer_column`: Column name for answers (default: `answer`)
* **Output:**

  * `mistral_jsonl`: Path to Mistral JSONL file

---

### **3. Training Mistral LLM with PEFT (LoRA)**

* Component: `Train Mistral LLM with PEFT (LoRA)`
* **Inputs:**

  * `jsonl_path`: Mistral JSONL file (from previous step)
  * `model_name`: HuggingFace Mistral model name (e.g. `mistralai/Mistral-7B-Instruct-v0.2`)
  * `output_dir`: Path to save the PEFT adapter/model (e.g. `/data/peft_lora_model`)
* **Output:**

  * `peft_model_dir`: Path to PEFT adapter/model directory

---

### **4. Merging LoRA Adapter into Base Model**

* Component: `Merge LoRA Adapter into Base Model`
* **Inputs:**

  * `base_model_name`: Base Mistral model (e.g. `mistralai/Mistral-7B-Instruct-v0.2`)
  * `peft_model_dir`: Directory of PEFT adapter (from training step)
  * `merged_model_dir`: Output directory for merged model (e.g. `/data/merged_model`)
* **Output:**

  * `merged_output_dir`: Directory containing the full, merged model

---

## **Directory Example**

```
/data/
  qna.csv
  mistral.jsonl
  peft_lora_model/
  merged_model/
```

---

## **Best Practices & Tips**

* **Always map the output of one node as the input to the next** in Elyra’s UI.
* **Set all required parameters and double-check paths.**
* Use a small model (e.g. `sshleifer/tiny-gpt2`) for test runs to save time and compute.
* All YAMLs are designed for simple, robust integration—no optional fields required.

---

## **License**

MIT

---

**For questions, raise an issue or contact the project owner.**

---
