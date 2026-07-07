"""
OPTIONAL: fine-tune a tiny DistilBERT classifier (via LoRA) to replace
RouterAgent's keyword heuristic with a learned "simple" vs "complex"
query classifier. This is what backs the "Fine-tuning" line on a GenAI
resume for this project — it's kept separate from the main app so the
core service stays lightweight and doesn't require torch/transformers
just to serve requests.

Not wired into the live API by default. To use it:
  1. pip install torch transformers peft datasets scikit-learn
  2. Label a small dataset of (question, label) pairs — 100-300 examples
     is enough for a demo. label is 0 = simple, 1 = complex.
  3. Run this script to produce a LoRA adapter under ./router_lora/
  4. Load it in RouterAgent.run() instead of the keyword heuristic.

Run:
    python scripts/finetune_router.py --data router_training_data.csv
"""
import argparse
import csv


def load_dataset(csv_path):
    questions, labels = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row["question"])
            labels.append(int(row["label"]))
    return questions, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with columns: question,label")
    parser.add_argument("--output", default="./router_lora", help="Output adapter dir")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    # Imported here, not at module load, so importing this file doesn't
    # force torch/transformers as a hard dependency of the whole project.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from sklearn.model_selection import train_test_split
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    questions, labels = load_dataset(args.data)
    train_q, val_q, train_l, val_l = train_test_split(
        questions, labels, test_size=0.2, random_state=42
    )

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

    train_ds = Dataset.from_dict({"text": train_q, "label": train_l}).map(tokenize, batched=True)
    val_ds = Dataset.from_dict({"text": val_q, "label": val_l}).map(tokenize, batched=True)

    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],  # DistilBERT attention projections
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./router_lora_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        learning_rate=2e-4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"LoRA router adapter saved to {args.output}")


if __name__ == "__main__":
    main()
