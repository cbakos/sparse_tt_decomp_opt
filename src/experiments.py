import wandb

if __name__ == '__main__':
    matrix_names = ["ex3", "ex10", "ex10hs", "ex13", "ex15"]
    method_names = ["factor", "amd", "rcm", "padding", "partial_gauss"]
    # note: factor can be done alone, but every other method uses it too
    # amd and rcm work on fixed/original n only
    # padding and partial_gauss: need to specify which n's to try - to get full picture just try "all"
    # i.e. for padding, do n, n+1, ... , 2n and for partial gauss do k = 0, 1, ..., n

    # second generation: combine two at a time:

    # experiment data columns:  tile size (int), variable_order (None, amd, rcm), padding (int), partial_gauss (int),
    #                           matrix_name (str), rank (int), max_mode_size (int)

    wandb.init(
        # set the wandb project where this run will be logged
        project="sparse_tt_decomp_opt",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        }
    )
