import math
from typing import List

import numpy as np
import scipy as sp
import scipy.sparse as ssp
from primePy import primes
from itertools import chain, combinations


def prime_factors(n: int) -> List[int]:
    """
    Returns the prime factors of n (including multiplicities).
    :param n: non-negative integer to be factored
    :return: prime factors of n
    """
    factors = primes.factors(n)
    return factors


def possible_tile_sizes_from_factors(factors: List[int]) -> np.array:
    """
    Returns the possible tile sizes from factors.
    :param factors: prime factors of n
    :return: unique tile sizes constructed as products of factors as a np.aray of size (k,)
    """

    factor_combinations = chain.from_iterable(combinations(factors, r) for r in range(1, len(factors) + 1))
    products = np.unique([math.prod(t) for t in factor_combinations])
    return products


def get_rank_from_tile_size(a: ssp.spmatrix, tile_size: int) -> int:
    """
    Returns the TTM-rank when a given tile size is used in the matrix2mpo algorithm.
    :param a: sparse, square matrix
    :param tile_size: size of sub-matrices used to cover the matrix, must be a divisor of the matrix size
    :return: TTM-rank when a given tile size is used in the matrix2mpo algorithm.
    """
    n = a.shape[0]
    r = 0  # the total maximal TTM-rank
    num_blocks = n // tile_size
    rows = [None] * num_blocks

    if not ssp.isspmatrix_csr(a):
        a = a.tocsr()

    for i in range(num_blocks):  # i counts the columns
        # determines which rows on this 'block column' are nonzero - linear indices
        nonzero_blocks_i = np.nonzero(np.sum(np.abs(a[:, i * tile_size:(i + 1) * tile_size]).toarray(), axis=1))[0]

        # if there is at least one nonzero element in this block column, then proceed
        if nonzero_blocks_i.size > 0:
            # round down to nearest integer - get row block index for each nonzero entry in the column block
            nonzero_blocks_i = np.floor_divide(nonzero_blocks_i, tile_size)

            # diff(I): finds transition indices from one block to the next
            # np.where: gives linear indices of these transition points
            # +1: gives index of the next block index
            # I[0]: gives first nonzero block index, others determined by previously created indices
            rows[i] = np.concatenate(([nonzero_blocks_i[0]],
                                      nonzero_blocks_i[np.where(np.diff(nonzero_blocks_i))[0] + 1]))

            # increment rank by the number of nonzero blocks in this column
            r += len(rows[i])
    return r
