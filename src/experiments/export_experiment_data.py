import wandb
import pandas as pd

if __name__ == '__main__':
    # Login to wandb (use your API key)
    wandb.login()

    # Specify the sweep ID
    sweep_id = 'cbakos/sparse_tt_decomp_opt/u372jlao'  # Replace with sweep ID

    # Initialize the WandB API
    api = wandb.Api()

    # Fetch the sweep using the sweep ID
    sweep = api.sweep(sweep_id)

    # Extract the runs from the sweep
    runs = sweep.runs

    # Collect the relevant data from each run
    all_data = []
    for run in runs:
        history = run.history()
        config = run.config
        for _, row in history.iterrows():
            row_data = {**config, **row, 'run_id': run.id, 'run_name': run.name}
            all_data.append(row_data)

    # Convert the data to a Pandas DataFrame
    df = pd.DataFrame(all_data)

    # Display the DataFrame
    print(df)

    # Save the DataFrame to a CSV file if needed
    df.to_csv('../../data/sweep_data.csv', index=False)
