from scipy.io import mmread
import scipy.sparse as ssp

import wandb
from optimizers.padding import sparse_padding
from optimizers.partial_gauss import partial_row_reduce
from optimizers.tile_size import prime_factors, possible_tile_sizes_from_factors, get_rank_from_tile_size
from optimizers.variable_ordering import amd_order, rcm_order


def run_experiments():
    # Initialize a new run
    wandb.init()
    # Access the parameters through wandb.config
    cfg = wandb.config

    path = "../../data/{}/{}.mtx".format(cfg.matrix_name, cfg.matrix_name)
    a = mmread(path)  # reads to coo_matrix format

    # minimize fill in
    if cfg.amd:
        order = amd_order(a)
        # perform row and column permutations
        a = a.tocsr()
        a.indices = order.take(a.indices)
        a = a.tocsc()
        a.indices = order.take(a.indices)
        a = a.tocsr()

    # reduce n
    if cfg.partial_gauss > 0:
        a = a.toarray()
        full_a = partial_row_reduce(a, cfg.partial_gauss)
        # get remaining part
        a = full_a[cfg.partial_gauss:, cfg.partial_gauss:]
        a = ssp.csr_matrix(a)

    # increase n
    if cfg.padding > 0:
        a = sparse_padding(a.tocoo(), cfg.padding)

    # concentrate nnz entries
    if cfg.rcm:
        order = rcm_order(a)
        a = a.tocsr()
        a.indices = order.take(a.indices)
        a = a.tocsc()
        a.indices = order.take(a.indices)

    # determine ranks, mode sizes and r2I6
    n = a.shape[0]
    z = a.nnz
    factors = prime_factors(n)
    max_mode_size = max(factors)
    tile_sizes = possible_tile_sizes_from_factors(factors)
    for tile in tile_sizes:
        r, _ = get_rank_from_tile_size(a, tile)

        # since we combine factors, maximum mode size is the max of the largest factor and chosen tile size
        max_mode_size = max(max_mode_size, tile)
        wandb.log({"rank": r, "max_mode_size": max_mode_size, "tile_size": tile, "z": z, "n": n})

