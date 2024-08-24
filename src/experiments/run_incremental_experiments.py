import numpy as np
from scipy.io import mmread

import wandb
import multiprocessing

from experiments.experiment_modules import amd_module, partial_gauss_module, rcm_module, get_sparsity
from optimizers.partial_gauss import partial_row_reduce_step
from src.optimizers.padding import sparse_padding
from src.optimizers.tile_size import prime_factors, possible_tile_sizes_from_factors, get_rank_from_tile_size
import scipy.sparse as ssp


def run_incremental_experiments():
    """
    Experiment script run by wandb agents.
    """
    # Initialize a new run
    wandb.init()
    # Access the parameters through wandb.config
    cfg = wandb.config

    path = "../../data/{}/{}.mtx".format(cfg.matrix_name, cfg.matrix_name)
    a = mmread(path)  # reads to coo_matrix format

    z_full, z_reduced = a.nnz, a.nnz
    n = a.shape[0]
    threshold = cfg.gauss_threshold

    # minimize fill in
    if cfg.amd:
        a = amd_module(a=a.tocsr())

    a_full = a.toarray()

    # Elimination process
    num_variables = int(cfg.partial_gauss*n)  # convert from fraction to number of variables
    pg_increments = cfg.partial_gauss_increments
    for i in range(0, num_variables, pg_increments):  # first k rows
        a_full = partial_row_reduce_step(a_full, i, n)

        # Use np.where to set values close to zero, to zero
        a_full = np.where(np.abs(a_full) < threshold, 0, a_full)

        # count nonzero entries of full (partially row-reduced) matrix
        z_full = np.count_nonzero(a_full)

        # get remaining part
        a = a_full[i:, i:]

        # number of nonzero entries in sub-matrix
        z_reduced = np.count_nonzero(a)
        new_n = a.shape[0]

        # if sparsity drops too low in remaining matrix, then quit run
        sparsity_ratio = get_sparsity(z_reduced, new_n)
        if sparsity_ratio < cfg.min_sparsity:
            wandb.log({"sparsity_ratio": sparsity_ratio})
            return  # no need to run any more experiments in this setup

        a = ssp.csr_matrix(a)

        # increase n
        if cfg.padding > 0:
            a = sparse_padding(a.tocoo(), cfg.padding)

        # concentrate nnz entries
        if cfg.rcm:
            a = rcm_module(a=a)

        # determine updated n, ranks, mode sizes
        new_n = a.shape[0]
        factors = prime_factors(new_n)
        max_mode_size = max(factors)
        tile_sizes = possible_tile_sizes_from_factors(factors)
        for tile in tile_sizes:
            r, _ = get_rank_from_tile_size(a, tile)

            # since we combine factors, maximum mode size is the max of the largest factor and chosen tile size
            max_mode_size = max(max_mode_size, tile)
            wandb.log({"rank": r,
                       "num_reduced_variables": i,
                       "max_mode_size": max_mode_size,
                       "tile_size": tile,
                       "z_full": z_full,
                       "z_reduced": z_reduced,
                       "n": new_n,
                       "sparsity_ratio": sparsity_ratio})


def run_agent(sweep_id):
    wandb.agent(sweep_id, function=run_incremental_experiments)


if __name__ == '__main__':
    num_agents = 4  # Number of parallel agents
    sweep_id = "cbakos/sparse_tt_decomp_opt/w2rr8teo"

    processes = []
    for _ in range(num_agents):
        process = multiprocessing.Process(target=run_agent, args=(sweep_id,))
        process.start()
        processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.join()
