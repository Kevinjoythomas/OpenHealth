"""Production training script for OpenHealth Doctor model.

Converted from models/train.ipynb. All hyperparameters are CLI-overridable.
Dataset lineage is written to models/lineage.json after loading.

Usage:
    python train.py
    python train.py --max-steps 500 --output-dir ./outputs
    python train.py --push-to-hub --hf-model kevinjoythomas/Llama3-Doctor
    python train.py --verify-dataset   # only computes dataset checksum, no training

Environment variables:
    WANDB_API_KEY   Set this to enable W&B logging without interactive prompt.
                    Leave unset to disable W&B entirely.
    HF_TOKEN        HuggingFace token for push-to-hub.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LINEAGE_PATH = Path(__file__).parent / "lineage.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune LLaMA-3 Doctor model with LoRA via Unsloth")
    p.add_argument("--base-model", default="unsloth/llama-3-8b-Instruct-bnb-4bit")
    p.add_argument("--hf-model", default="kevinjoythomas/Llama3-Doctor",
                   help="HuggingFace repo to save/push the fine-tuned model")
    p.add_argument("--dataset", default="kevinjoythomas/Doctor-Dataset")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--push-to-hub", action="store_true",
                   help="Push fine-tuned model to HuggingFace Hub after training")
    p.add_argument("--verify-dataset", action="store_true",
                   help="Load dataset, compute checksum, update lineage.json, then exit")
    return p.parse_args()


def _dataset_checksum(dataset) -> str:
    """Compute a SHA-256 fingerprint over all prompt fields."""
    h = hashlib.sha256()
    for row in dataset:
        h.update(row["prompt"].encode())
    return h.hexdigest()


def update_lineage(args: argparse.Namespace, row_count: int, checksum: str) -> None:
    lineage = json.loads(LINEAGE_PATH.read_text()) if LINEAGE_PATH.exists() else {}
    lineage["dataset"] = {
        **lineage.get("dataset", {}),
        "name": args.dataset,
        "row_count": row_count,
        "sha256_prompts": checksum,
    }
    lineage["training"] = {
        **lineage.get("training", {}),
        "max_steps": args.max_steps,
        "per_device_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "seed": args.seed,
    }
    lineage["logged_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    LINEAGE_PATH.write_text(json.dumps(lineage, indent=2))
    log.info("Lineage written to %s", LINEAGE_PATH)


def main() -> None:
    args = parse_args()

    # Disable W&B if key not set — avoids interactive prompt in CI
    if not os.getenv("WANDB_API_KEY"):
        os.environ["WANDB_DISABLED"] = "true"
        log.info("WANDB_API_KEY not set — W&B logging disabled")

    from datasets import load_dataset

    log.info("Loading dataset %s ...", args.dataset)
    dataset_train = load_dataset(args.dataset, split="train")
    row_count = len(dataset_train)
    log.info("Dataset loaded: %d rows", row_count)

    log.info("Computing dataset checksum (SHA-256 over prompt field)...")
    checksum = _dataset_checksum(dataset_train)
    log.info("Dataset checksum: %s", checksum)

    update_lineage(args, row_count, checksum)

    if args.verify_dataset:
        log.info("--verify-dataset: done. Exiting without training.")
        sys.exit(0)

    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer

    log.info("Loading base model %s ...", args.base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model=model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=args.seed,
        use_rslora=False,
        use_dora=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        dataset_text_field="prompt",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=5,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=args.output_dir,
            report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        ),
    )

    log.info("Starting training: max_steps=%d effective_batch=%d", args.max_steps, args.batch_size * args.grad_accum)
    trainer_stats = trainer.train()
    log.info("Training complete: %s", trainer_stats)

    log.info("Saving model to %s ...", args.hf_model)
    model.save_pretrained(args.hf_model)

    if args.push_to_hub:
        log.info("Pushing to HuggingFace Hub: %s", args.hf_model)
        model.push_to_hub(args.hf_model, tokenizer=tokenizer)
        log.info("Push complete.")


if __name__ == "__main__":
    main()
