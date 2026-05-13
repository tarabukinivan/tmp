#!/usr/bin/env python3
"""
train_full_ft_single.py -- Full fine-tune of Moonlight-16B on a SINGLE GPU.

For B300 (288GB) / B200 (180GB) / H200 (141GB) instances. Simpler than
DeepSpeed for single-GPU: native PyTorch + optional 8-bit AdamW.

Memory budget for 16B BF16 model on single GPU:
  - Model params:       32 GB (BF16)
  - Gradients:          32 GB (BF16)
  - Optimizer states:   128 GB (FP32 AdamW) OR 32 GB (8-bit AdamW)
  - Activations:        ~20-30 GB (batch=2, seq=2048, checkpointing)
  - Total FP32 AdamW:   ~215 GB → fits B300 (288), tight on B200 (180)
  - Total 8-bit AdamW:  ~115 GB → fits all (H200, B200, B300)

Usage:
  python scripts/train_full_ft_single.py \\
      --base-model /root/Moonlight-16B-A3B-Instruct \\
      --dataset /root/sft_train.jsonl \\
      --val-dataset /root/sft_val.jsonl \\
      --output /root/distil-full-ft \\
      --max-seq-len 2048 \\
      --batch-size 2 \\
      --grad-accum 16 \\
      --epochs 2 \\
      --lr 5e-6 \\
      --optimizer adamw_8bit   # or adamw (default if B300)
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

# transformers 5.x compat shims
try:
    import transformers.cache_utils as _cu
    if not hasattr(_cu, "HybridCache"):
        _cu.HybridCache = _cu.DynamicCache  # type: ignore[attr-defined]
except Exception:
    pass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


class SFTDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_seq_len: int, limit: int = 0):
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.records = []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = r.get("messages") or []
                if len(msgs) < 2 or msgs[0]["role"] != "user":
                    continue
                self.records.append(r)
        if not self.records:
            raise ValueError(f"No valid records in {path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        msgs = r["messages"]
        try:
            prompt_text = self.tok.apply_chat_template(
                msgs[:1], tokenize=False, add_generation_prompt=True)
            full_text = self.tok.apply_chat_template(
                msgs[:2], tokenize=False, add_generation_prompt=False)
        except Exception:
            prompt_text = msgs[0]["content"] + "\n"
            full_text = prompt_text + msgs[1]["content"]
        if not full_text.startswith(prompt_text):
            full_text = prompt_text + msgs[1]["content"]
        prompt_ids = self.tok(prompt_text, add_special_tokens=False).input_ids
        full_ids = self.tok(full_text, add_special_tokens=False).input_ids
        full_ids = full_ids[: self.max_seq_len]
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = list(full_ids)
        for i in range(prompt_len):
            labels[i] = -100
        return {"input_ids": full_ids, "labels": labels}


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    ids, labels, mask = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = max_len - n
        ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        mask.append([1] * n + [0] * pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model",       required=True, dest="base_model")
    ap.add_argument("--dataset",          required=True)
    ap.add_argument("--val-dataset",      default=None, dest="val_dataset")
    ap.add_argument("--output",           required=True)
    ap.add_argument("--max-seq-len",      type=int, default=2048, dest="max_seq_len")
    ap.add_argument("--batch-size",       type=int, default=2, dest="batch_size",
                    help="per-step batch size (before grad_accum)")
    ap.add_argument("--grad-accum",       type=int, default=16, dest="grad_accum")
    ap.add_argument("--epochs",           type=int, default=2)
    ap.add_argument("--lr",               type=float, default=5e-6)
    ap.add_argument("--warmup-ratio",     type=float, default=0.03, dest="warmup_ratio")
    ap.add_argument("--weight-decay",     type=float, default=0.01, dest="weight_decay")
    ap.add_argument("--max-grad-norm",    type=float, default=1.0, dest="max_grad_norm")
    ap.add_argument("--optimizer",        default="adamw",
                    choices=["adamw", "adamw_8bit"],
                    help="adamw=full FP32 (needs B300), adamw_8bit=bitsandbytes")
    ap.add_argument("--save-every",       type=int, default=500, dest="save_every")
    ap.add_argument("--eval-every",       type=int, default=500, dest="eval_every")
    ap.add_argument("--log-every",        type=int, default=10, dest="log_every")
    ap.add_argument("--seed",             type=int, default=42)
    ap.add_argument("--limit",            type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)

    log("="*72)
    log(f"Full fine-tune (single GPU)")
    log(f"  base:       {args.base_model}")
    log(f"  dataset:    {args.dataset}")
    log(f"  val:        {args.val_dataset}")
    log(f"  output:     {args.output}")
    log(f"  seq_len:    {args.max_seq_len}")
    log(f"  batch:      {args.batch_size} × grad_accum {args.grad_accum} "
        f"= effective {args.batch_size*args.grad_accum}")
    log(f"  epochs:     {args.epochs}, lr={args.lr}, optimizer={args.optimizer}")
    log("="*72)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    log("Loading dataset...")
    train_ds = SFTDataset(args.dataset, tokenizer, args.max_seq_len, limit=args.limit)
    log(f"  train: {len(train_ds)} examples")
    val_ds = None
    if args.val_dataset:
        val_ds = SFTDataset(args.val_dataset, tokenizer, args.max_seq_len)
        log(f"  val:   {len(val_ds)} examples")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    log("Loading model (BF16)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2",
        )
    except Exception as e:
        log(f"  flash_attention_2 unavailable ({e}), falling back to default")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.cuda()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  total params: {n_params:,}")
    log(f"  VRAM after load: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Optimizer
    log(f"Building optimizer: {args.optimizer}")
    if args.optimizer == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.PagedAdamW8bit(
                model.parameters(), lr=args.lr,
                betas=(0.9, 0.95), eps=1e-8,
                weight_decay=args.weight_decay,
            )
        except ImportError:
            log("  bitsandbytes not installed; falling back to AdamW FP32")
            optimizer = AdamW(model.parameters(), lr=args.lr,
                              betas=(0.9, 0.95), eps=1e-8,
                              weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr,
                          betas=(0.9, 0.95), eps=1e-8,
                          weight_decay=args.weight_decay)
    log(f"  VRAM after optimizer: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    log(f"  steps/epoch: {steps_per_epoch}, total: {total_steps}, warmup: {warmup_steps}")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0
    best_val_loss = float("inf")
    running_loss = 0.0
    running_n = 0

    for epoch in range(1, args.epochs + 1):
        log(f"--- Epoch {epoch}/{args.epochs} ---")
        epoch_t0 = time.time()
        accum_count = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss / args.grad_accum
            loss.backward()
            running_loss += outputs.loss.item()
            running_n += 1
            accum_count += 1

            if accum_count >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                global_step += 1

                if global_step % args.log_every == 0:
                    avg = running_loss / running_n
                    lr_now = scheduler.get_last_lr()[0]
                    vram = torch.cuda.memory_allocated() / 1e9
                    log(f"E{epoch} step {global_step:>5}/{total_steps} | "
                        f"loss={avg:.4f} | lr={lr_now:.2e} | vram={vram:.1f}GB")
                    running_loss = 0.0; running_n = 0

                # Save checkpoint
                if global_step % args.save_every == 0:
                    save_path = out_dir / f"checkpoint-{global_step}"
                    log(f"Saving {save_path}")
                    model.save_pretrained(str(save_path), safe_serialization=True)
                    tokenizer.save_pretrained(str(save_path))
                    log(f"  saved.")

                # Eval
                if val_ds is not None and global_step % args.eval_every == 0:
                    val_loss = run_eval(model, val_ds, tokenizer, args)
                    log(f"  val_loss={val_loss:.4f}  best={best_val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = out_dir / "best"
                        log(f"  *** BEST (saving {best_path}) ***")
                        model.save_pretrained(str(best_path), safe_serialization=True)
                        tokenizer.save_pretrained(str(best_path))
                    model.train()

        epoch_t = time.time() - epoch_t0
        log(f"Epoch {epoch} done in {epoch_t/60:.1f} min")

    # Final save
    final_path = out_dir / f"epoch{args.epochs}_final"
    log(f"Saving final → {final_path}")
    model.save_pretrained(str(final_path), safe_serialization=True)
    tokenizer.save_pretrained(str(final_path))
    log(f"Training complete. best_val_loss: {best_val_loss:.4f}")


def run_eval(model, val_ds, tokenizer, args):
    model.eval()
    loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )
    total = 0.0; n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            total += outputs.loss.item()
            n += 1
    return total / max(n, 1)


if __name__ == "__main__":
    main()
