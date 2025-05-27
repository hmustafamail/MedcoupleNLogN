"""
Support and standalone functions for Medcouple robust measure of skewness in N*log(N) time for Python 3.12

Validated against Statsmodels implementation (N^2 time)

Authors: Guy Brys (c. 2004), Jordi Gutiérrez Hermoso (c. 2015), Mustafa I. Hussain (2025)

References
----------

Guy Brys, Mia Hubert and Anja Struyf (2004) A Robust Measure of Skewness; JCGS 13 (4), 996-1017. 
"""

import numpy as np
from statsmodels.tools.sm_exceptions import ValueWarning
import pdb


def _medcouple_1d_legacy(y):
    """
    Calculates the medcouple robust measure of skew. Less efficient version of
    the algorithm which computes in O(N**2) time. Useful for validating the
    O(N log N) version and for applications requiring legacy behavior.

    Parameters
    ----------
    y : array_like, 1-d
        Data to compute use in the estimator.

    Returns
    -------
    mc : float
        The medcouple statistic

    Notes
    -----
    This version of the algorithm requires a O(N**2) memory allocations, and so may
    not work for very large arrays (N>10000).

    .. [*] M. Hubert and E. Vandervieren, "An adjusted boxplot for skewed
       distributions" Computational Statistics & Data Analysis, vol. 52, pp.
       5186-5201, August 2008.
    """

    # Parameter changes the algorithm to the slower for large n

    y = np.squeeze(np.asarray(y))
    if y.ndim != 1:
        raise ValueError("y must be squeezable to a 1-d array")

    y = np.sort(y)

    n = y.shape[0]
    if n % 2 == 0:
        mf = (y[n // 2 - 1] + y[n // 2]) / 2
    else:
        mf = y[(n - 1) // 2]

    z = y - mf
    lower = z[z <= 0.0]
    upper = z[z >= 0.0]
    upper = upper[:, None]
    standardization = upper - lower
    is_zero = np.logical_and(lower == 0.0, upper == 0.0)
    standardization[is_zero] = np.inf
    spread = upper + lower
    h = spread / standardization
    # GH5395
    num_ties = np.sum(lower == 0.0)
    if num_ties:

        # Replacements has -1 above the anti-diagonal, 0 on the anti-diagonal,
        # and 1 below the anti-diagonal
        replacements = np.ones((num_ties, num_ties)) - np.eye(num_ties)
        replacements -= 2 * np.triu(replacements)

        # Convert diagonal to anti-diagonal
        replacements = np.fliplr(replacements)

        # Always replace upper right block
        h[:num_ties, -num_ties:] = replacements

    return np.median(h)


def _signum(x):
    r"""
    Sign function that returns -1, 0, or 1 based on the input.

    Parameters
    ----------
    x : int
        A signed integer value.

    Returns
    -------
    int
        -1 if x < 0, 0 if x == 0, 1 if x > 0.

    Notes
    -----
    This function is used in the fast medcouple implementation to
    handle tie-breaking when two values are numerically close.
    """
    return int(x > 0) - int(x < 0)


def _wmedian(A, W):
    r"""
    Compute the weighted median of the values in A using the associated weights in W.

    Parameters
    ----------
    A : 1-d NumPy array of float
        The numeric values for which the weighted median is to be computed.
    W : 1-d NumPy array of int
        The corresponding non-negative integer weights for each value in A.

    Returns
    -------
    float
        The weighted median of A. If there are multiple medians due to tied weights,
        the lower median is returned.

    Notes
    -----
    This is a helper function for the O(N log N) medcouple algorithm.
    """

    # Validation: NaN protection
    if np.any(np.isnan(A)) or np.any(np.isnan(W)):
        raise ValueError("A and W may not contain NaN.")

    # Ensure 1-d arrays
    if A.ndim != 1 or W.ndim != 1:
        raise ValueError("A and W must be 1-dimensional arrays.")

    # Ensure same length
    if A.shape[0] != W.shape[0]:
        raise ValueError("A and W must have the same length.")

    # Sort A and W according to A
    idx = np.argsort(A)
    A_sorted = A[idx]
    W_sorted = W[idx]

    # Compute cumulative sum of weights
    w_cumsum = np.cumsum(W_sorted)
    wtot = w_cumsum[-1]

    # Find the smallest index i such that cumulative weight >= total weight / 2
    median_idx = np.searchsorted(w_cumsum, wtot / 2, side="left")

    return A_sorted[median_idx]


def _construct_A_W(L, R, Zplus, Zminus, n_plus, eps2):
    """
    Vectorized construction of A and W as NumPy arrays.

    Parameters
    ----------
    L : np.ndarray
        1-d array of left bounds.
    R : np.ndarray
        1-d array of right bounds.
    Zplus : np.ndarray
        1-d array of input values.
    Zminus :  np.ndarray
        1-d array of input values.
    n_plus : int
    eps2 : float

    Returns
    -------
    A : np.ndarray
        Array of kernel values.
    W : np.ndarray
        Corresponding weights.
    valid_i : np.ndarray
        Indices used for construction.
    """
    if not isinstance(L, np.ndarray) or not isinstance(R, np.ndarray):
        raise ValueError("L and R must be np.ndarrays")

    L = np.asarray(L) # TODO: remove
    R = np.asarray(R) # this too

    valid_i = np.where(L <= R)[0]
    L_valid = L[valid_i]
    R_valid = R[valid_i]
    mid_indices = (L_valid + R_valid) // 2
    
    A = np.empty_like(valid_i, dtype=float)
    for k in range(valid_i.size):
        A[k] = _h_kern(valid_i[k], mid_indices[k], Zplus, Zminus, n_plus, eps2)

    W = R_valid - L_valid + 1
    return A, W, valid_i


def _h_kern(index_plus, index_minus, Zplus, Zminus, n_plus, eps2):
    """
    H kernel function.

    Parameters
    ----------
    index_plus : int-like
        Index of Zplus.
    index_minus : int-like
        Index of Zminus.
    Zplus : np.ndarray
        1-d array of input values.
    Zminus :  np.ndarray
        1-d array of input values.
    n_plus : int
    eps2 : float

    Returns
    -------
    float
    """
    if not np.isscalar(index_plus) or not np.isscalar(index_minus):
        raise ValueError("index_plus and index_minus must be ints")

    zp_i = Zplus[index_plus]
    zm_i = Zminus[index_minus]
    
    if abs(zp_i - zm_i) <= 2 * eps2:
        return _signum(n_plus - 1 - index_plus - index_minus)
    return (zp_i + zm_i) / (zp_i - zm_i)


def _finalize_h_kernel_sweep(L, R, Zplus, Zminus, n_plus, eps2):
    """
    Compute the final array A.

    Parameters
    ----------
    L : np.ndarray
        1-d array of left indices.
    R : np.ndarray
        1-d array of right indices.
    Zplus : np.ndarray
        1-d array of input values.
    Zminus :  np.ndarray
        1-d array of input values.
    n_plus : int
    eps2 : float

    Returns
    -------
    A : numpy.ndarray of float
        1-d array of sorted h_kern values in descending order.
    """
    L = np.asarray(L)
    R = np.asarray(R)

    # Determine total number of h_kern values.
    total_count = int(np.sum(R - L + 1))
    
    # Preallocate an array for the results.
    A = np.empty(total_count, dtype=np.float64)
    
    # Position to insert next block of values.
    pos = 0
    
    # Loop over each index_plus element.
    for i in range(L.shape[0]):
        left = L[i]
        right = R[i]
        
        # Loop over each corresponding index_minus.
        for j in range(left, right + 1):

            # Here both i and j are scalars.
            A[pos] = _h_kern(i, j, Zplus, Zminus, n_plus, eps2)
            pos += 1

    # Sort in descending order and return.
    return np.sort(A)[::-1]


def _medcouple_nlogn(X, eps1=2**-52, eps2=2**-1022):
    r"""
    Calculates the medcouple robust measure of skewness. Faster version of the
    algorithm which computes in O(N log N) time.

    Parameters
    ----------
    X : np.ndarray
        Input 1-d array of numeric values.

    Returns
    -------
    float
        The medcouple statistic.

    Notes
    -----

    NaNs are not automatically removed. If present in the input, the result
    will be NaN.

    .. [*] Guy Brys, Mia Hubert and Anja Struyf (2004) A Robust Measure
       of Skewness; JCGS 13 (4), 996-1017.
    """

    if np.any(np.isnan(X)):
        return np.nan

    n = len(X)

    if n < 3:
        from warnings import warn
        msg = (
            "medcouple is undefined for input with less than 3 elements. "
            "Returning NaN."
        )
        warn(msg, ValueWarning)
        return np.nan

    if n < 10:
        from warnings import warn
        msg = (
            "Fast medcouple algorithm (use_fast=True) is not recommended "
            "for small datasets (N < 10). Results may be unstable. Consider "
            "using use_fast=False for accuracy."
        )
        warn(msg, UserWarning)

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

    # Zplus, Zminus are 1-d np.ndarrays
    Zplus = Z[Z >= -Zeps]
    Zminus = Z[Z <= Zeps]

    # get lengths
    n_plus = Zplus.shape[0]
    n_minus = Zminus.shape[0]

    # construct L, R as numpy arrays
    L = np.zeros(n_plus, dtype=int)
    R = np.full(n_plus, n_minus - 1, dtype=int)

    Ltot = 0
    Rtot = n_minus * n_plus
    medc_idx = Rtot // 2

    while Rtot - Ltot > n_plus:

        # Construct A, W as NumPy arrays
        A, W, _ = _construct_A_W(L, R, Zplus, Zminus, n_plus, eps2)

        h_med = _wmedian(A, W)

        Am_eps = eps1 * (eps1 + np.abs(h_med))

        # Preallocate arrays P and Q of length n_plus.
        P = np.empty(n_plus, dtype=int)
        Q = np.empty(n_plus, dtype=int)

        # Construct P. Note: We traverse i in reversed order.
        j = 0
        for idx in range(n_plus):

            # i goes in reversed order; use reversed indices.
            i = n_plus - 1 - idx

            # Increase j until the condition is no longer met.
            while j < n_minus and _h_kern(i, j, Zplus, Zminus, n_plus, eps2) - h_med > Am_eps:
                j += 1

            # j-1 is our current value for that i.
            # Store it in P at the reversed index; we will fix the order later.
            P[idx] = j - 1

        # Reverse P to get the correct order.
        P = P[::-1]

        # Construct Q.
        j = n_minus - 1
        for i in range(n_plus):
            while j >= 0 and _h_kern(i, j, Zplus, Zminus, n_plus, eps2) - h_med < -Am_eps:
                j -= 1
            Q[i] = j + 1

        # Compute sumP and sumQ.
        sumP = np.sum(P) + n_plus
        sumQ = np.sum(Q)

        if medc_idx <= sumP - 1:
            R = P
            Rtot = sumP
        elif medc_idx > sumQ - 1:
            L = Q
            Ltot = sumQ
        else:
            return h_med

    A = _finalize_h_kernel_sweep(L, R, Zplus, Zminus, n_plus, eps2)
    return A[medc_idx - Ltot]


def _medcouple_1d(y, use_fast=True):
    """
    Calculates the medcouple robust measure of skew.

    Parameters
    ----------
    y : array_like, 1-d
        Data to compute use in the estimator.
    use_fast : bool
        Whether to use the O(n log n) implementation. Defaults to True.

    Returns
    -------
    mc : float
        The medcouple statistic
    """
    y = np.squeeze(np.asarray(y))
    if y.ndim != 1:
        raise ValueError("y must be squeezable to a 1-d array")

    if use_fast:
        return _medcouple_nlogn(y)
    else:
        return _medcouple_1d_legacy(y)


def medcouple(y, axis=0, use_fast=True):
    """
    Calculate the medcouple robust measure of skew.

    Parameters
    ----------
    y : array_like
        Data to compute use in the estimator.
    axis : {int, None}
        Axis along which the medcouple statistic is computed.  If `None`, the
        entire array is used.
    use_fast : bool
        Whether to use the faster O(N log N) implementation. Default is True.
        To use the legacy O(N**2) version, set to False.

    Returns
    -------
    mc : float or ndarray
        The medcouple statistic.

    Notes
    -----
    The legacy algorithm (``use_fast=False``) uses an O(N**2) implementation
    which provides exact results and is reliable for all dataset sizes,
    including small inputs and cases with ties. However, it requires a O(N**2)
    memory allocations, and so may not work for very large arrays (N>10000).

    The fast algorithm (``use_fast=True``) implements an O(N log N)
    approximation which is optimized for large datasets. **It is not intended
    for small sample sizes (N < 10)** or datasets with a high proportion of
    ties, as it may yield numerically unstable or inaccurate results in these
    cases. For such inputs, prefer ``use_fast=False`` to ensure correctness.

    If NaNs are present in the input when use_fast=True, the result will be
    NaN. To preserve legacy behavior, a number may be returned when
    use_fast=False.

    If the size of ``y`` is less than 3 and ``use_fast=True``, the result will
    be NaN. To preserve legacy behavior, a value may be returned when
    ``use_fast=False``.

    Small numerical differences are possible based on the choice of algorithm.

    .. [*] Guy Brys, Mia Hubert and Anja Struyf (2004) A Robust Measure
       of Skewness; JCGS 13 (4), 996-1017.

    .. [*] M. Hubert and E. Vandervieren, "An adjusted boxplot for skewed
       distributions" Computational Statistics & Data Analysis, vol. 52, pp.
       5186-5201, August 2008.
    """
    y = np.asarray(y, dtype=np.double)  # GH 4243
    if axis is None:
        return _medcouple_1d(y.ravel(), use_fast=use_fast)

    return np.apply_along_axis(_medcouple_1d, axis, y, use_fast=use_fast)
