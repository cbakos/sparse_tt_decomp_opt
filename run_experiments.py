from scipy.io import mmread
import scipy.sparse as ssp

import wandb
import multiprocessing
from src.optimizers.padding import sparse_padding
from src.optimizers.partial_gauss import partial_row_reduce
from src.optimizers.tile_size import prime_factors, possible_tile_sizes_from_factors, get_rank_from_tile_size
from src.optimizers.variable_ordering import amd_order, rcm_order


def run_experiments():
    """
    Experiment script run by wandb agents.
    """
    # Initialize a new run
    wandb.init()
    # Access the parameters through wandb.config
    cfg = wandb.config

    path = "data/{}/{}.mtx".format(cfg.matrix_name, cfg.matrix_name)
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


def run_agent(sweep_id):
    wandb.agent(sweep_id, function=run_experiments)


if __name__ == '__main__':
    num_agents = 6  # Number of parallel agents
    sweep_id = "cbakos/sparse_tt_decomp_opt/xlo8p9vq"

    processes = []
    for _ in range(num_agents):
        process = multiprocessing.Process(target=run_agent, args=(sweep_id,))
        process.start()
        processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.join()


