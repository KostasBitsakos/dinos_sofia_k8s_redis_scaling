"""Small numpy-only statistics for the gates: Student t CDF/quantile, paired one-sided test, Holm correction."""
import math

import numpy as np


def _betacf(a, b, x, itmax=300, eps=3e-16):
    fpmin = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > fpmin else fpmin)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > fpmin else fpmin)
        c = 1.0 + aa / c
        c = c if abs(c) > fpmin else fpmin
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > fpmin else fpmin)
        c = 1.0 + aa / c
        c = c if abs(c) > fpmin else fpmin
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    x = df / (df + t * t)
    p = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - p if t > 0 else p


def t_ppf(q, df):
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paired_one_sided(diffs, alpha=0.05):
    """H0: mean(diff) <= 0. Returns mean, sd, one-sided p, and the (1 - alpha) one-sided lower bound
    (= lower end of the two-sided (1 - 2 alpha) CI; alpha = 0.025 gives the two-sided 95% CI lower bound)."""
    d = np.asarray(diffs, dtype=float)
    n = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if sd == 0.0:
        return dict(n=n, mean=mean, sd=0.0, p=0.0 if mean > 0 else 1.0, lb=mean)
    se = sd / math.sqrt(n)
    tstat = mean / se
    return dict(n=n, mean=mean, sd=sd, p=1.0 - t_cdf(tstat, n - 1), lb=mean - t_ppf(1 - alpha, n - 1) * se)


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def paired_ci(diffs, level=0.95):
    """Two-sided paired t CI of the mean difference: (mean, lo, hi)."""
    d = np.asarray(diffs, dtype=float)
    mean = float(d.mean())
    if d.size < 2 or float(d.std(ddof=1)) == 0.0:
        return mean, mean, mean
    half = t_ppf(0.5 + level / 2, d.size - 1) * float(d.std(ddof=1)) / math.sqrt(d.size)
    return mean, mean - half, mean + half
