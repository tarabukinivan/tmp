"""
GRPO training — NAVWORLD
GPU: A100 80GB (или multi-GPU если доступно)

Улучшения по сравнению с предыдущей версией:
  1. Гибридный reward: реальный Affine score + rule-based fallback
  2. Динамический датасет: промпты генерируются из grpo_data.jsonl
     но можно переключить на af.GAME().get_task() для полностью живых задач
  3. Поддержка multi-GPU через device_map="auto"
  4. Лучшие гиперпараметры для NAVWORLD

Для multi-GPU запуск:
    torchrun --nproc_per_node=N python grpo_train.py
"""

import torch
import json
import logging
import random

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from peft import LoraConfig, TaskType
import wandb

from grpo_reward import get_reward_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── Конфигурация датасета ─────────────────────────────────────────
MAX_EXAMPLES = 2000
MIN_SCORE    = 0.15
SEED         = 42

# ── Конфигурация reward ───────────────────────────────────────────
# True  = гибридный (Affine + rule-based) — лучшее качество обучения
# False = только rule-based — быстрее, для отладки
USE_AFFINE_REWARD = True


def load_grpo_dataset(data_path: str = "grpo_data.jsonl") -> Dataset:
    """
    Загружает датасет с стратифицированной выборкой по score.

    Стратегия (взята у veritas-rl): нам нужна вариация reward в батче.
    Берём равномерно из трёх бакетов: low/mid/high score.
    Это обеспечивает что в каждом батче есть примеры с разным reward.
    """
    examples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    examples = [e for e in examples if e["score"] >= MIN_SCORE]
    logger.info(f"После фильтра score>={MIN_SCORE}: {len(examples)} примеров")

    low  = [e for e in examples if e["score"] < 0.35]
    mid  = [e for e in examples if 0.35 <= e["score"] < 0.55]
    high = [e for e in examples if e["score"] >= 0.55]

    random.seed(SEED)
    per_bucket = MAX_EXAMPLES // 3
    sampled = (
        random.sample(low,  min(per_bucket, len(low)))  +
        random.sample(mid,  min(per_bucket, len(mid)))  +
        random.sample(high, min(per_bucket, len(high)))
    )
    random.shuffle(sampled)

    scores = [e["score"] for e in sampled]
    logger.info(
        f"Датасет: {len(sampled)} примеров | "
        f"min={min(scores):.3f} max={max(scores):.3f} avg={sum(scores)/len(scores):.3f}"
    )
    logger.info(f"Бакеты: low={len(low)} mid={len(mid)} high={len(high)}")
    return Dataset.from_list(sampled)


def get_lora_config(rank: int = 16) -> LoraConfig:
    """
    LoRA конфигурация.

    rank=16 для GRPO (был 8) — больше rank = больше expressiveness.
    Veritas-RL использует full fine-tune на 4B модели,
    у нас 32B поэтому LoRA r=16 — разумный компромисс.
    """
    return LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # Все projection layers для максимального охвата
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )


def train_grpo(
    model_name: str = "./sft_model_navworld/merged",
    data_path: str = "grpo_data.jsonl",
    output_dir: str = "./grpo_model_navworld",
    num_epochs: int = 1,
):
    wandb.init(
        project="affine-sft-grpo",
        name="grpo-navworld-v2",
        config={
            "model": model_name,
            "use_affine_reward": USE_AFFINE_REWARD,
            "max_examples": MAX_EXAMPLES,
            "min_score": MIN_SCORE,
        }
    )

    # ── Токенайзер ───────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Модель ───────────────────────────────────────────────────
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",        # автоматически на все доступные GPU
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # требует flash-attn
    )
    logger.info(f"Model loaded on: {[p.device for p in list(model.parameters())[:1]]}")

    # ── LoRA ─────────────────────────────────────────────────────
    lora_config = get_lora_config(rank=16)

    # ── Датасет ──────────────────────────────────────────────────
    dataset = load_grpo_dataset(data_path)

    steps_per_epoch = len(dataset) // 4  # grad_accum=4
    logger.info(
        f"Шагов: ~{steps_per_epoch} | "
        f"~{steps_per_epoch * 97 / 3600:.1f} часов при 97 сек/шаг"
    )

    # ── GRPOConfig ───────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,

        # Батч и аккумуляция
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,   # effective batch = 4

        # Оптимизация
        learning_rate=5e-7,   # Немного ниже чем раньше — стабильнее для GRPO
        optim="adamw_torch_fused",
        bf16=True,

        # Генерация completions
        max_completion_length=512,
        num_generations=4,    # 4 вместо 2 — больше контраста для GRPO
                               # При 4×A100 можно поднять до 8
        temperature=0.8,      # Чуть выше для разнообразия

        # KL penalty — не даёт модели слишком далеко уйти от base
        # Важно для стабильности при реальном Affine reward
        beta=0.04,

        # Сохранение
        save_steps=100,
        save_total_limit=3,

        # Логирование
        logging_steps=5,
        report_to=["wandb"],

        # Память
        dataloader_pin_memory=True,
    )

    # ── Reward function ──────────────────────────────────────────
    logger.info(
        f"Reward mode: {'Hybrid (Affine + rule-based)' if USE_AFFINE_REWARD else 'Rule-based only'}"
    )
    reward_fn = get_reward_fn(use_affine=USE_AFFINE_REWARD)

    # ── Trainer ──────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    logger.info("Starting GRPO training...")
    trainer.train()

    # ── Сохранение ───────────────────────────────────────────────
    best_dir = f"{output_dir}/best"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    logger.info(f"✅ Done! Saved to {best_dir}")

    return trainer.model, tokenizer


if __name__ == "__main__":
    train_grpo()
