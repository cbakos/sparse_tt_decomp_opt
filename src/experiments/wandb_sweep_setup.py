import wandb
import yaml

if __name__ == '__main__':
    """
    After adjusting config.yaml, run this script to initialize wandb sweep. Copy the returned sweep_id.
    Then go to the terminal (e.g. Cmder on Windows) and start up agents (separate terminal windows) to process the jobs:
    wandb agent cbakos/sparse_tt_decomp_opt/sweep_id  # add generated sweep id
    """
    # Load the YAML configuration file
    with open('src/experiments/config.yaml', 'r') as file:
        sweep_config = yaml.safe_load(file)

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project='sparse_tt_decomp_opt')
