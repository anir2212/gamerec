"""Assignment, SRM, power, CUPED, fixed-horizon inference and peeking simulation."""

from __future__ import annotations

import hashlib
import math
import random
from statistics import NormalDist

import numpy as np
from scipy.stats import chisquare, norm


def assign(unit_id: str, salt: str = "gamerec-v1") -> str:
    digest = hashlib.sha256(f"{salt}:{unit_id}".encode()).digest()
    return "A" if int.from_bytes(digest[:8], "big") % 2 == 0 else "B"


def srm(count_a: int, count_b: int, expected_a: float = 0.5) -> dict:
    if min(count_a, count_b) < 0 or count_a + count_b == 0 or not 0 < expected_a < 1:
        raise ValueError("invalid assignment counts or expected allocation")
    total = count_a + count_b
    statistic, p = chisquare([count_a, count_b], f_exp=[total * expected_a, total * (1 - expected_a)])
    return {"chi_square": float(statistic), "p_value": float(p), "flag_0_001": bool(p < 0.001)}


def sample_size_per_arm(p0: float, absolute_lift: float, alpha: float = 0.05,
                        power: float = 0.8) -> int:
    p1 = p0 + absolute_lift
    if not (0 < p0 < 1 and 0 < p1 < 1 and 0 < alpha < 1 and 0 < power < 1):
        raise ValueError("invalid probabilities")
    pooled = (p0 + p1) / 2
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    numerator = z_alpha * math.sqrt(2 * pooled * (1 - pooled)) + z_power * math.sqrt(
        p0 * (1 - p0) + p1 * (1 - p1))
    return math.ceil((numerator / absolute_lift) ** 2)


def cuped(outcomes: list[float], preperiod: list[float]) -> tuple[list[float], float]:
    if len(outcomes) != len(preperiod) or len(outcomes) < 2:
        raise ValueError("paired arrays need at least two observations")
    x, y = np.asarray(preperiod, dtype=float), np.asarray(outcomes, dtype=float)
    variance = float(np.var(x))
    theta = float(np.cov(x, y, ddof=0)[0, 1] / variance) if variance else 0.0
    adjusted = y - theta * (x - np.mean(x))
    return adjusted.tolist(), theta


def binary_result(a: list[int], b: list[int], bootstrap: int = 1000,
                  seed: int = 17) -> dict:
    if not a or not b or any(value not in (0, 1) for value in a + b):
        raise ValueError("nonempty binary outcomes required")
    pa, pb = sum(a) / len(a), sum(b) / len(b)
    pooled = (sum(a) + sum(b)) / (len(a) + len(b))
    null_se = math.sqrt(pooled * (1 - pooled) * (1 / len(a) + 1 / len(b)))
    z = (pb - pa) / null_se if null_se else 0.0
    p_value = 2 * norm.sf(abs(z)) if null_se else 1.0
    rng = random.Random(seed)
    differences = []
    for _ in range(bootstrap):
        aa = sum(rng.choices(a, k=len(a))) / len(a)
        bb = sum(rng.choices(b, k=len(b))) / len(b)
        differences.append(bb - aa)
    differences.sort()
    return {"control_n": len(a), "treatment_n": len(b), "control_rate": pa,
            "treatment_rate": pb, "difference": pb - pa,
            "p_value_two_sided": float(p_value),
            "bootstrap_ci95": [differences[int(0.025 * bootstrap)],
                               differences[min(bootstrap - 1, int(0.975 * bootstrap))]],
            "srm": srm(len(a), len(b))}


def peeking_false_positive(seed: int = 7, repetitions: int = 500,
                           days: int = 10, daily_per_arm: int = 100) -> dict:
    """Under a true null, compare daily uncorrected looks to one fixed final look."""
    rng = np.random.default_rng(seed)
    fixed_flags, peek_flags = 0, 0
    for _ in range(repetitions):
        a = rng.binomial(1, 0.2, size=days * daily_per_arm)
        b = rng.binomial(1, 0.2, size=days * daily_per_arm)
        peeked = False
        for day in range(1, days + 1):
            end = day * daily_per_arm
            pooled = (a[:end].sum() + b[:end].sum()) / (2 * end)
            se = math.sqrt(2 * pooled * (1 - pooled) / end)
            z = (b[:end].mean() - a[:end].mean()) / se if se else 0
            peeked |= 2 * norm.sf(abs(z)) < 0.05
            if day == days:
                fixed_flags += 2 * norm.sf(abs(z)) < 0.05
        peek_flags += peeked
    return {"seed": seed, "repetitions": repetitions, "looks": days,
            "fixed_horizon_false_positive_rate": fixed_flags / repetitions,
            "daily_peeking_false_positive_rate": peek_flags / repetitions}
