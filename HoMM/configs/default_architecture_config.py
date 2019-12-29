import tensorflow as tf

default_architecture_config = {
   "input_shape": [4],  # dimension(s) of base input
   "output_shape": [1],  # dimension(s) of base output

   "outcome_shape": None,  # If not None, allows outcomes != targets, for RL etc.
   "output_masking": False,  # If True, enables output masking, for RL tasks

   "separate_target_network": False,  # construct a separate network for e.g. Q-learning targets

   "IO_num_hidden": 128,  # number of hidden units in input/output nets.

   "z_dim": 512,  # latent dimensionality

   "M_num_hidden_layers": 3,  # meta-network hidden layers
   "M_num_hidden": 512,  # meta-network hidden units
   "M_max_pool": True,  # if False, average pools.

   "H_num_hidden_layers": 3,  # hyper network hidden layers
   "H_num_hidden": 512,  # hyper network hidden units
   "task_weight_weight_mult": 1.0,  # weight scale init for hyper network to
                                    # approximately preserve variance in task
                                    # network

   "F_num_hidden_layers": 3,  # task network num hidden layers
   "F_num_hidden": 64,  # task network hidden units

   "F_weight_normalization": False,  # if True, weight vectors in task network
                                     # will be normalized to unit length --
                                     # Jeff Clune says this plus predicting
                                     # magnitude separately helps in this kind
                                     # of context.
   "F_wn_strategy": "standard",  # one of "standard", "unit_until_last", 
                                 # "all_unit", which respectively fit scalar
                                 # weight magnitudes for all, only last layer,
                                 # and none of the task network weights. 
                                 # Use "standard" for the basic weight-norm
                                 # approach.

   "internal_nonlinearity": tf.nn.leaky_relu,  # nonlinearity for hidden layers
                                               # -- note that output to Z is
                                               # linear by default

   "persistent_task_reps": False,  # enable persistent task representations
                                   # -- not included in original paper, but
                                   # can be useful for learning complex tasks

   "train_drop_prob": 0.0,  # drop prob for task representations 

   "lang_drop_prob": 0.0,  # drop prob for language processing network
   "num_lstm_layers": 2,  # for language processing
   "max_sentence_len": 20,  # for language

   "optimizer": "Adam",  # Adam or RMSProp are supported options
   "memory_buffer_size": 1024,  # how many data points to remember (per task)
   "meta_batch_size": 50, # how many points the meta-learner sees each step
}
