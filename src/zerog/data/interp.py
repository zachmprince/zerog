"""ENDF-style interpolation of tabulated functions."""

import math

import numpy as np
from numba import njit


@njit(cache=True, error_model="numpy")
def evaluate_tabulated(x, y, breakpoints, interpolation, query):
    """Evaluate a TAB1 function at each value in query, holding end values outside its range.

    x must be ascending, as stored in the library. breakpoints holds the 1-based index ending each
    interpolation region, and interpolation its ENDF rule: 1 histogram, 2 lin-lin, 3 lin-log,
    4 log-lin, 5 log-log.
    """
    out = np.empty(query.shape[0])
    for n in range(query.shape[0]):
        q = query[n]
        if q <= x[0]:
            out[n] = y[0]
            continue
        if q >= x[-1]:
            out[n] = y[-1]
            continue
        i = np.searchsorted(x, q, side="right") - 1
        region = 0
        while i >= breakpoints[region] - 1:
            region += 1
        rule = interpolation[region]
        if rule == 1:
            out[n] = y[i]
        elif rule == 2:
            out[n] = y[i] + (q - x[i]) / (x[i + 1] - x[i]) * (y[i + 1] - y[i])
        elif rule == 3:
            out[n] = y[i] + math.log(q / x[i]) / math.log(x[i + 1] / x[i]) * (y[i + 1] - y[i])
        elif rule == 4:
            out[n] = y[i] * math.exp((q - x[i]) / (x[i + 1] - x[i]) * math.log(y[i + 1] / y[i]))
        elif rule == 5:
            frac = math.log(q / x[i]) / math.log(x[i + 1] / x[i])
            out[n] = y[i] * math.exp(frac * math.log(y[i + 1] / y[i]))
        else:
            raise ValueError("unknown interpolation rule")
    return out
