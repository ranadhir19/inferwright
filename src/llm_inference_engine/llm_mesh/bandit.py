"""llm_mesh.bandit: self-improving routing.

``BanditRouter`` keeps one UCB1 bandit per task type over the registry's
models. Each scored outcome (observed quality) updates the arm statistics,
so routing converges toward the best model per task without any prior
knowledge of the quality profiles. ``DynamicSelector`` is a simpler
history-based alternative: it scores models from rolling observed means
of quality, cost, and latency.
"""

import math
from typing import Dict, List, Tuple

from .models import TASK_TYPES, ModelRegistry


class BanditRouter:
    """UCB1 bandit per task type; reward = observed quality in [0, 1]."""

    def __init__(self, registry: ModelRegistry, c: float = 1.0):
        self.registry = registry
        self.c = c
        self.counts: Dict[Tuple[str, str], int] = {}
        self.sums: Dict[Tuple[str, str], float] = {}
        self.n_requests = 0
        self.history: List[Dict] = []  # per-request: task, model, reward, regret

    def _arms(self, task_type: str) -> List[str]:
        return [n for n in self.registry.models if n not in self.registry.down]

    def oracle_quality(self, task_type: str) -> float:
        arms = self._arms(task_type)
        return max(self.registry.quality_of(n, task_type) for n in arms)

    def oracle_model(self, task_type: str) -> str:
        arms = self._arms(task_type)
        return max(arms, key=lambda n: self.registry.quality_of(n, task_type))

    def select(self, task_type: str) -> str:
        arms = self._arms(task_type)
        if not arms:
            raise RuntimeError("no models available")
        total = sum(self.counts.get((task_type, a), 0) for a in arms)
        # play each arm once first (deterministic order)
        for a in arms:
            if self.counts.get((task_type, a), 0) == 0:
                return a
        best, best_ucb = arms[0], -1e18
        for a in arms:
            n = self.counts[(task_type, a)]
            mean = self.sums[(task_type, a)] / n
            ucb = mean + self.c * math.sqrt(math.log(total) / n)
            if ucb > best_ucb:
                best, best_ucb = a, ucb
        return best

    def update(self, task_type: str, model: str, reward: float) -> None:
        key = (task_type, model)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.sums[key] = self.sums.get(key, 0.0) + reward
        self.n_requests += 1
        regret = self.oracle_quality(task_type) - self.registry.quality_of(
            model, task_type)
        self.history.append({"task": task_type, "model": model,
                             "reward": reward, "regret": regret})

    def mean_reward(self, task_type: str, model: str) -> float:
        n = self.counts.get((task_type, model), 0)
        return self.sums.get((task_type, model), 0.0) / n if n else 0.0

    def selection_share(self, task_type: str) -> Dict[str, float]:
        total = sum(self.counts.get((task_type, a), 0)
                    for a in self._arms(task_type))
        if total == 0:
            return {}
        return {a: self.counts.get((task_type, a), 0) / total
                for a in self._arms(task_type)}


class DynamicSelector:
    """Pick per request from rolling observed quality/cost/latency means."""

    def __init__(self, registry: ModelRegistry, w_cost: float = 0.6,
                 w_lat: float = 0.4, window: int = 200):
        self.registry = registry
        self.w_cost = w_cost
        self.w_lat = w_lat
        self.window = window
        self.obs: Dict[Tuple[str, str], List[Tuple[float, float, float]]] = {}

    def observe(self, task_type: str, model: str, quality: float,
                cost_usd: float, latency_ms: float) -> None:
        key = (task_type, model)
        buf = self.obs.setdefault(key, [])
        buf.append((quality, cost_usd, latency_ms))
        if len(buf) > self.window:
            del buf[0]

    def select(self, task_type: str, in_tokens: int, out_tokens: int) -> str:
        arms = [n for n in self.registry.models
                if n not in self.registry.down]
        if not arms:
            raise RuntimeError("no models available")
        stats = {}
        for a in arms:
            buf = self.obs.get((task_type, a), [])
            if buf:
                q = sum(x[0] for x in buf) / len(buf)
                c = sum(x[1] for x in buf) / len(buf)
                l = sum(x[2] for x in buf) / len(buf)
            else:  # no observations yet: fall back to analytical profile
                q = self.registry.quality_of(a, task_type)
                c = self.registry.expected_cost_usd(a, in_tokens, out_tokens)
                l = self.registry.expected_latency_ms(a, out_tokens)
            stats[a] = (q, c, l)
        qs = {a: s[0] for a, s in stats.items()}
        cs = {a: s[1] for a, s in stats.items()}
        ls = {a: s[2] for a, s in stats.items()}

        def norm(d):
            lo, hi = min(d.values()), max(d.values())
            return {k: 0.0 if hi == lo else (v - lo) / (hi - lo)
                    for k, v in d.items()}

        cn, ln = norm(cs), norm(ls)
        return max(arms,
                   key=lambda a: qs[a] - self.w_cost * cn[a]
                   - self.w_lat * ln[a])
