default_architecture_config = {
input_shape: [4],
output_shape: [1],
num_task_hidden_layers: 3,
M_num_hidden_layers: 3,
H_num_hidden_layers: 3,
F_num_hidden: 64,
z_dim: 512,
H_num_hidden: 512,
M_num_hidden: 512,
IO_num_hidden: 128,
task_weight_weight_mult: 1.0,
persistent_task_reps: True,
output_nonlinearity: None,
lang_drop_prob: 0.0,
train_drop_prob: 0.0,
num_lstm_layers: 2,
optimizer: Adam,
max_sentence_len: 20,
memory_buffer_size: 1024,
}