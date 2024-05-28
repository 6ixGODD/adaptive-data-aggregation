from collections import namedtuple

import numpy as np


# n_bits = 4;
# max_q = (2 ** (n_bits -1)) -1;
# min_q = -2 ** (n_bits -1);

# print(n_bits, min_q, max_q)

def quantize(a, n_bits):
    # quantize a into integer with n_bits
    max_q = (2 ** (n_bits - 1)) - 1
    min_q = -2 ** (n_bits - 1)
    # if np.isnan(a):
    #     return 0
    if a > max_q:
        return max_q
    if a < min_q:
        return min_q
    return int(a)
