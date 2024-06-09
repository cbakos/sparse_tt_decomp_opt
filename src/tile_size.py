import math
from typing import List, Tuple

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


def get_rank_from_tile_size(a: ssp.spmatrix, tile_size: int) -> Tuple[int, List[np.array]]:
    """
    Returns the TTM-rank and nonzero block locations, when a given tile size is used in the matrix2mpo algorithm.
    :param a: sparse, square matrix
    :param tile_size: size of sub-matrices used to cover the matrix, must be a divisor of the matrix size
    :return: TTM-rank, nonzero block locations: rows[i] contains the nonzero row-block indices for column-block i
    """
    n = a.shape[0]
    r = 0  # the total maximal TTM-rank
    num_blocks = n // tile_size
    rows = [[]] * num_blocks

    if not ssp.isspmatrix_csr(a):
        a = a.tocsr()

    for i in range(num_blocks):  # i counts the columns-blocks
        # determines which rows on this 'block column' are nonzero - linear indices
        col_i_nonzero_row_indices = np.nonzero(np.sum(np.abs(a[:, i * tile_size:(i + 1) * tile_size]).toarray(), axis=1))[0]

        # if there is at least one nonzero element in this block column, then proceed
        if col_i_nonzero_row_indices.size > 0:
            # round down to nearest integer - get row block index for each nonzero entry in the column block
            col_i_nonzero_row_block_indices = np.floor_divide(col_i_nonzero_row_indices, tile_size)

            # np.diff: finds transition points from one block to the next
            block_transition_diff_mask = np.diff(col_i_nonzero_row_block_indices)
            # np.where: gives linear indices from these transition difference values (+1 since 0th element is skipped)
            block_transition_indices = np.where(block_transition_diff_mask)[0] + 1

            first_nonzero_row_block_index = [col_i_nonzero_row_block_indices[0]]
            remaining_nonzero_row_block_indices = col_i_nonzero_row_block_indices[block_transition_indices]
            rows[i] = np.concatenate((first_nonzero_row_block_index, remaining_nonzero_row_block_indices))

            # increment rank by the number of nonzero blocks in this column
            r += len(rows[i])
    return r, rows
