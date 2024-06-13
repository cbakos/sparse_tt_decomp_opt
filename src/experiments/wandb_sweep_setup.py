import wandb
import yaml

if __name__ == '__main__':
    # Load the YAML configuration file
    with open('config.yaml', 'r') as file:
        sweep_config = yaml.safe_load(file)

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project='sparse_tt_decomp_opt')
    # Start the sweep agent
    # wandb.agent(sweep_id, function=run_experiments)
