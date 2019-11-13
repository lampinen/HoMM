import tensorflow as tf

default_architecture_config = {
   "input_shape": [4],
   "output_shape": [1],

   "outcome_shape": None,  # If not None, allows outcomes != targets, for RL etc.
   "output_masking": False,  # If True, enables output masking, for RL tasks

   "IO_num_hidden": 128,

   "z_dim": 512,

   "M_num_hidden_layers": 3,
   "M_num_hidden": 512,
   "M_max_pool": True,  # if False, average pools.

   "H_num_hidden_layers": 3,
   "H_num_hidden": 512,
   "task_weight_weight_mult": 1.0,

   "F_num_hidden_layers": 3,
   "F_num_hidden": 64,

   "internal_nonlinearity": tf.nn.leaky_relu, 
   "output_nonlinearity": None,

   "persistent_task_reps": False,

   "lang_drop_prob": 0.0,
   "train_drop_prob": 0.0,

   "num_lstm_layers": 2,

   "optimizer": "Adam",
   "max_sentence_len": 20,
   "memory_buffer_size": 1024,
   "meta_batch_size": 50, # how many points the meta-learner sees
}
