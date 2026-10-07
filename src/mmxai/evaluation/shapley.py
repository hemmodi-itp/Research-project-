"""Exact Shapley values over modality subsets (Sec. 3.3, Table 5)."""

from itertools import combinations
from math import factorial


def shapley_values(players, value):
    """phi_m = sum_{S subset N\\{m}} |S|! (n-|S|-1)! / n! * (v(S + m) - v(S)).

    `value` maps a frozenset of players to a number and must define the empty set. With n = 3 modalities this
    needs the 8 subsets; the values sum to v(N) - v(empty) (efficiency).
    """
    players = tuple(players)
    n = len(players)
    phi = {}
    for m in players:
        others = [p for p in players if p != m]
        total = 0.0
        for r in range(len(others) + 1):
            weight = factorial(r) * factorial(n - r - 1) / factorial(n)
            for S in combinations(others, r):
                total += weight * (value[frozenset(S) | {m}] - value[frozenset(S)])
        phi[m] = total
    return phi
