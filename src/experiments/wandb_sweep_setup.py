import wandb
import yaml

if __name__ == '__main__':
    """
    After adjusting config.yaml, run this script to initialize wandb sweep. Copy the returned sweep_id.
    Next, go and configure src/experiments/run_incremental_experiments.py before running it.
    """
    # Load the YAML configuration file
    with open('config.yaml', 'r') as file:
        sweep_config = yaml.safe_load(file)

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project='sparse_tt_decomp_opt')
