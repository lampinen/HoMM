default_run_config = {
    "train_base": True,  # whether to train base meta-learning 
    "train_meta": True,  # whether to train meta class/map tasks, or just base
    "train_language_base": False,  # whether to train language -> base tasks 
    "train_language_meta": False,  # whether to train language -> meta tasks
    
    "num_epochs": 10000,  # number of training epochs
    "refresh_mem_buffs_every": 50, # how many epochs between updates to buffers
    "eval_every": 20,  # how many epochs between evals

    "num_runs": 5, # number of independent runs of the network to complete

    "init_learning_rate": 3e-5,  # initial learning rate for base tasks
    "init_language_learning_rate": 3e-5,  # for language-cued tasks
    "init_meta_learning_rate": 1e-5,  # for meta-classification and mappings

    "lr_decay": 0.85,  # how fast base task lr decays (multiplicative)
    "language_lr_decay": 0.8, 
    "meta_lr_decay": 0.85,
    "lr_decays_every": 100,  # lr decays happen once per this many epochs

    "min_learning_rate": 3e-8,  # can't decay past these minimum values 
    "min_language_learning_rate": 3e-8,
    "min_meta_learning_rate": 1e-7,
}
