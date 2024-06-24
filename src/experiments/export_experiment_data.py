import wandb
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_run_history(run):
    history = run.history()
    config = run.config
    run_data = []
    for _, row in history.iterrows():
        row_data = {**config, **row, 'run_id': run.id, 'run_name': run.name}
        run_data.append(row_data)
    return run_data


if __name__ == '__main__':
    # Login to wandb (use your API key)
    wandb.login()

    # Specify the sweep ID
    sweep_id = 'cbakos/sparse_tt_decomp_opt/oqzu6g7c'  # Replace with sweep ID

    # Initialize the WandB API
    api = wandb.Api()

    # Fetch the sweep using the sweep ID
    sweep = api.sweep(sweep_id)

    # Extract the runs from the sweep
    runs = sweep.runs

    # Use ThreadPoolExecutor to fetch run histories in parallel
    all_data = []
    num_workers = 100
    with ThreadPoolExecutor(max_workers=num_workers) as executor:  # Adjust max_workers as needed
        futures = {executor.submit(fetch_run_history, run): run for run in runs}
        for future in tqdm(as_completed(futures), total=len(futures)):
            run_data = future.result()
            all_data.extend(run_data)

    # Convert the data to a Pandas DataFrame
    df = pd.DataFrame(all_data)

    # Display the DataFrame
    print(df)

    # Save the DataFrame to a CSV file if needed
    df.to_csv('../../data/sweep_0_10_data1.csv', index=False)
