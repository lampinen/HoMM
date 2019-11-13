default_run_config = {
    "train_language": False,
    "train_meta": True,
    
    "num_epochs": 10000,  # number of training epochs
    "refresh_mem_buffs_every": 50, # how many epochs between updates to buffers
    "eval_every": 20,  # how many epochs between evals

    "num_runs": 5,

    "init_learning_rate": 3e-5,
    "init_language_learning_rate": 3e-5,
    "init_meta_learning_rate": 1e-5,

    "lr_decay": 0.85,
    "language_lr_decay": 0.8,
    "meta_lr_decay": 0.85,
    "lr_decays_every": 100,

    "min_learning_rate": 3e-8,
    "min_language_learning_rate": 3e-8,
    "min_meta_learning_rate": 1e-7,
}
