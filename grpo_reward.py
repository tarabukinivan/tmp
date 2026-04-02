"""
GRPO reward function для NAVWORLD.

Стратегия: гибридный reward
  - Быстрый rule-based (всегда, мгновенно)
  - Реальный Affine score (параллельно, через af SDK)

Идея взята у veritas-rl: verifiable reward лучше rule-based.
Для NAVWORLD нет точного правильного ответа, поэтому используем
реальный Affine score как основной сигнал + rule-based как fallback.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Маркеры для rule-based scoring ───────────────────────────────
REQUIRED_TOOLS = [
    "poi_search", "weather", "direction",
    "around_search", "search_flights", "search_train_tickets"
]
ANSWER_MARKERS = [
    "##", "行程", "方案", "预算", "推荐", "景点", "餐厅",
    "酒店", "交通", "住宿", "Day", "第一天", "第二天", "第三天"
]

# Веса гибридного reward
AFFINE_WEIGHT = 0.7
RULE_WEIGHT   = 0.3


# ─────────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────────

def extract_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                c = item.get("content", "")
                if c:
                    parts.append(str(c))
            elif isinstance(item, (str, int, float)):
                parts.append(str(item))
        return " ".join(parts)
    return str(completion)


def rule_based_score(text: str) -> float:
    if not text or len(text.strip()) < 10:
        return 0.05
    reward = 0.1
    tools_found = sum(1 for t in REQUIRED_TOOLS if t in text)
    reward += min(tools_found / 3.0, 1.0) * 0.4
    markers_found = sum(1 for m in ANSWER_MARKERS if m in text)
    reward += min(markers_found / 4.0, 1.0) * 0.3
    length = len(text)
    if 200 <= length <= 1500:
        reward += 0.2
    elif length < 200:
        reward += length / 200 * 0.2
    else:
        reward += max(0.0, 0.2 - (length - 1500) / 5000 * 0.2)
    return min(reward, 1.0)


# ─────────────────────────────────────────────────────────────────
# Affine scorer (singleton)
# ─────────────────────────────────────────────────────────────────

class AffineScorer:
    """
    Singleton для оценки completions через Affine SDK.
    Использует кэш чтобы не переоценивать одинаковые completion+task.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._ready = False
        return cls._instance

    def __init__(self):
        if self._ready:
            return
        self._ready = True
        self._cache = {}
        self._env = None
        try:
            import affine as af
            self._env = af.GAME()
            logger.info("✅ Affine GAME environment ready")
        except Exception as e:
            logger.warning(f"⚠️  Affine init failed ({e}). Rule-based only.")

    def score_one(self, task_id: int, text: str) -> Optional[float]:
        if self._env is None:
            return None
        key = (task_id, hash(text[:200]))
        if key in self._cache:
            return self._cache[key]
        try:
            task = self._env.get_task(task_id=task_id)
            raw = self._env.score(task, text)
            val = float(raw) / 100.0
            self._cache[key] = val
            return val
        except Exception as e:
            logger.debug(f"Affine score failed task={task_id}: {e}")
            return None

    def score_batch(self, task_ids: List[int], texts: List[str],
                    timeout: float = 25.0) -> List[Optional[float]]:
        if self._env is None:
            return [None] * len(task_ids)
        results = [None] * len(task_ids)

        def _score(idx, tid, txt):
            results[idx] = self.score_one(tid, txt)

        with ThreadPoolExecutor(max_workers=min(8, len(task_ids))) as ex:
            futs = [ex.submit(_score, i, t, txt)
                    for i, (t, txt) in enumerate(zip(task_ids, texts))]
            for f in futs:
                try:
                    f.result(timeout=timeout)
                except Exception:
                    pass
        return results


# ─────────────────────────────────────────────────────────────────
# Reward function
# ─────────────────────────────────────────────────────────────────

def get_reward_fn(use_affine: bool = True):
    """
    Гибридная reward function для trl 0.29.1 GRPOTrainer.

    use_affine=True  → пробуем реальный Affine score + rule-based
    use_affine=False → только rule-based (для отладки/скорости)
    """
    scorer = AffineScorer() if use_affine else None

    def reward_fn(
        prompts: List,
        completions,
        task_id: List[int] = None,
        score: List[float] = None,
        **kwargs
    ) -> List[float]:

        n = len(completions)
        texts = [extract_text(c) for c in completions]

        # Rule-based (мгновенно)
        rule_scores = [rule_based_score(t) for t in texts]

        # Affine (параллельно)
        affine_scores = [None] * n
        if use_affine and scorer is not None and task_id is not None:
            try:
                affine_scores = scorer.score_batch(
                    task_ids=list(task_id),
                    texts=texts,
                    timeout=25.0
                )
            except Exception as e:
                logger.warning(f"Affine batch failed: {e}")

        # Комбинируем
        final = []
        n_affine = 0
        for i in range(n):
            a = affine_scores[i]
            r = rule_scores[i]
            if a is not None:
                final.append(min(AFFINE_WEIGHT * a + RULE_WEIGHT * r, 1.0))
                n_affine += 1
            else:
                final.append(r)

        avg = sum(final) / n
        std = (sum((x - avg) ** 2 for x in final) / n) ** 0.5
        logger.info(
            f"Rewards: min={min(final):.3f} max={max(final):.3f} "
            f"avg={avg:.3f} std={std:.3f} "
            f"[affine={n_affine}/{n}]"
        )
        return final

    return reward_fn
