from scipy.io import mmread

import wandb
import multiprocessing

from experiments.experiment_modules import amd_module, partial_gauss_module, rcm_module
from src.optimizers.padding import sparse_padding
from src.optimizers.tile_size import prime_factors, possible_tile_sizes_from_factors, get_rank_from_tile_size


def run_experiments():
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

    # minimize fill in
    if cfg.amd:
        a = amd_module(a=a.tocsr())

    # reduce n
    if cfg.partial_gauss > 0:
        a, z_full, z_reduced = partial_gauss_module(a=a,
                                                    num_variables=cfg.partial_gauss,
                                                    threshold=cfg.gauss_threshold)

    # increase n
    # todo: move padding after rcm
    if cfg.padding > 0:
        a = sparse_padding(a.tocoo(), cfg.padding)

    # concentrate nnz entries
    if cfg.rcm:
        a = rcm_module(a=a)

    # determine ranks, mode sizes
    n = a.shape[0]
    factors = prime_factors(n)
    max_mode_size = max(factors)
    tile_sizes = possible_tile_sizes_from_factors(factors)
    for tile in tile_sizes:
        r, _ = get_rank_from_tile_size(a, tile)

        # since we combine factors, maximum mode size is the max of the largest factor and chosen tile size
        max_mode_size = max(max_mode_size, tile)
        wandb.log({"rank": r,
                   "max_mode_size": max_mode_size,
                   "tile_size": tile,
                   "z_full": z_full,
                   "z_reduced": z_reduced,
                   "n": n})


def run_agent(sweep_id):
    wandb.agent(sweep_id, function=run_experiments)


if __name__ == '__main__':
    num_agents = 12  # Number of parallel agents
    sweep_id = "cbakos/sparse_tt_decomp_opt/7sj1sv80"

    processes = []
    for _ in range(num_agents):
        process = multiprocessing.Process(target=run_agent, args=(sweep_id,))
        process.start()
        processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.join()
