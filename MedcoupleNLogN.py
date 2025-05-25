"""
Support and standalone functions for Medcouple robust measure of skewness in N*log(N) time for Python 3.12

Validated against Statsmodels implementation (N^2 time)

Authors: Guy Brys (c. 2004), Jordi Gutiérrez Hermoso (c. 2015), Mustafa I. Hussain (2025)

References
----------

Guy Brys, Mia Hubert and Anja Struyf (2004) A Robust Measure of Skewness; JCGS 13 (4), 996-1017. 
"""

import numpy as np
import warnings
from itertools import tee
from typing import List


def signum(x: int) -> int:
    return (x > 0) - (x < 0)


def wmedian(A: List[float], W: List[int]) -> float:
    """Compute the weighted median of A with corresponding weights W."""
    AW = sorted(zip(A, W), key=lambda x: x[0])
    wtot = sum(W)
    beg = 0
    end = len(AW) - 1

    while True:
        mid = (beg + end) // 2
        trial = AW[mid][0]

        wleft = sum(w for a, w in AW if a < trial)
        wright = sum(w for a, w in AW if a >= trial)

        if 2 * wleft > wtot:
            end = mid
        elif 2 * wright < wtot:
            beg = mid
        else:
            return trial


def medcouple_nlogn(X: np.ndarray, eps1: float = 2**-52, eps2: float = 2**-1022) -> float:
    """
    Calculates the medcouple robust measure of skewness. The implementation follows the algorithm described by Brys, Hubert, and Struyf (2004).

    Parameters
    ----------
    X : np.ndarray
        Input 1D array of numeric values.

    Returns
    -------
    float
        The medcouple statistic.
    """

    # Remove NaNs
    X = X[~np.isnan(X)]
    n = len(X)

    if n < 3:
        warnings.warn("medcouple is undefined for input with less than 3 elements.")
        return float('nan')

    Z = np.sort(X)[::-1]
    n2 = (n - 1) // 2
    Zmed = Z[n2] if n % 2 else (Z[n2] + Z[n2 + 1]) / 2

    if np.abs(Z[0] - Zmed) < eps1 * (eps1 + np.abs(Zmed)):
        return -1.0
    if np.abs(Z[-1] - Zmed) < eps1 * (eps1 + np.abs(Zmed)):
        return 1.0

    Z -= Zmed
    Zden = 2 * max(Z[0], -Z[-1])
    Z /= Zden
    Zmed /= Zden
    Zeps = eps1 * (eps1 + np.abs(Zmed))

    Zplus = Z[Z >= -Zeps]
    Zminus = Z[Z <= Zeps]
    n_plus = len(Zplus)
    n_minus = len(Zminus)

    def h_kern(i: int, j: int) -> float:
        a = Zplus[i]
        b = Zminus[j]
        if abs(a - b) <= 2 * eps2:
            return signum(n_plus - 1 - i - j)
        return (a + b) / (a - b)

    L = [0] * n_plus
    R = [n_minus - 1] * n_plus
    Ltot = 0
    Rtot = n_minus * n_plus
    medc_idx = Rtot // 2

    while Rtot - Ltot > n_plus:
        valid_i = [i for i in range(n_plus) if L[i] <= R[i]]
        I1, I2 = tee(valid_i)

        A = [h_kern(i, (L[i] + R[i]) // 2) for i in I1]
        W = [R[i] - L[i] + 1 for i in I2]
        h_med = wmedian(A, W)
        Am_eps = eps1 * (eps1 + abs(h_med))

        P, Q = [], []
        j = 0
        for i in reversed(range(n_plus)):
            while j < n_minus and h_kern(i, j) - h_med > Am_eps:
                j += 1
            P.append(j - 1)
        P.reverse()

        j = n_minus - 1
        for i in range(n_plus):
            while j >= 0 and h_kern(i, j) - h_med < -Am_eps:
                j -= 1
            Q.append(j + 1)

        sumP = sum(P) + len(P)
        sumQ = sum(Q)

        if medc_idx <= sumP - 1:
            R = P
            Rtot = sumP
        elif medc_idx > sumQ - 1:
            L = Q
            Ltot = sumQ
        else:
            return h_med

    A = []
    for i, (l, r) in enumerate(zip(L, R)):
        A.extend(h_kern(i, j) for j in range(l, r + 1))

    A.sort(reverse=True)
    return A[medc_idx - Ltot]
