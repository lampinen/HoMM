from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from copy import deepcopy
from itertools import permutations

from polynomials import *
from orthogonal_matrices import random_orthogonal

### Parameters #################################################

config = {
    "run_offset": 0,
    "num_runs": 5,

    "num_variables": 4,
    "max_degree": 2,

    "num_input": 4, # 4 variables
    "num_output": 1, # 1 output

    "num_hidden": 64,
    "num_hidden_hyper": 512,
    "num_hidden_language": 512,

    "init_learning_rate": 1e-4,
#    "init_language_learning_rate": 1e-4,
    "init_meta_learning_rate": 1e-4,

    "new_init_learning_rate": 1e-5,
#    "new_init_language_learning_rate": 1e-5,
    "new_init_meta_learning_rate": 1e-5,

    "lr_decay": 0.85,
#    "language_lr_decay": 0.8,
    "meta_lr_decay": 0.9,

    "lr_decays_every": 100,
    "min_learning_rate": 3e-8,
    "min_language_learning_rate": 1e-8,
    "min_meta_learning_rate": 3e-7,

    "refresh_meta_cache_every": 1, # how many epochs between updates to meta_cache
    "refresh_mem_buffs_every": 50, # how many epochs between updates to buffers

    "max_base_epochs": 10000,
    "max_new_epochs": 1000,
    "num_task_hidden_layers": 3,
    "num_hyper_hidden_layers": 3,
    "train_drop_prob": 0.00, # dropout probability, applied on meta and hyper
                             # but NOT task or input/output at present. Note
                             # that because of multiplicative effects and depth
                             # impact can be dramatic.

    "task_weight_weight_mult": 1., # not a typo, the init range of the final
                                   # hyper weights that generate the task
                                   # parameters. 

    "output_dir": "/mnt/fs2/lampinen/polynomials/basic/",
    "save_every": 20, 
    "sweep_meta_batch_sizes": [10, 20, 30, 50, 100, 200, 400, 800], # if not None,
                                                                    # eval each at
                                                                    # training ends

    "memory_buffer_size": 1024, # How many points for each polynomial are stored
    "meta_batch_size": 128, # how many meta-learner sees
    "early_stopping_thresh": 0.05,
    "num_base_tasks": 100, # prior to meta-augmentation
    "num_new_tasks": 10,

    "meta_add_vals": [-3, -2, -1, 1, 3],
    "meta_mult_vals": [-3, -1, 3],
    "num_meta_binary_pairss": 200, # for binary tasks like multiplying 
                                   # polynomials, how many pairs does the 
                                   # system see?
    "new_meta_tasks": [],
    "new_meta_mappings": ["permute_3210", "add_2", "add_-2", "mult_2", "mult_-2"],
    
#    "train_language": False, # whether to train language as well (only language
#                            # inputs, for now)
#    "lang_drop_prob": 0.0, # dropout on language processing features
#                            # to try to address overfitting

    "internal_nonlinearity": tf.nn.leaky_relu,
    "output_nonlinearity": None
}

poly_fam = polynomial_family(config["num_variables"], config["max_degree"])
config["variables"] = poly_fam.variables

config["base_meta_tasks"] = ["is_constant_polynomial"] + ["is_intercept_nonzero"] + ["is_%s_relevant" % var for var in config["variables"]]
config["base_meta_mappings"] = ["square"] + ["add_%f" % c for c in config["meta_add_vals"]] + ["mult_%f" % c for c in config["meta_mult_vals"]] + ["permute_" + "".join([str(x) for x in p]) for p in permutations(range(config["num_variables"]))]
config["base_meta_binary_funcs"] = ["binary_sum", "binary_mult"] 


# filtering out held-out meta tasks
config["base_meta_tasks"] = [x for x in config["base_meta_tasks"] if x not in config["new_meta_tasks"]]
config["meta_mappings"] = [x for x in config["base_meta_mappings"] if x not in config["new_meta_mappings"]]

config["base_tasks"] = [poly_fam.sample_polynomial() for _ in range(config["num_base_tasks"])]
config["new_tasks"] = [poly_fam.sample_polynomial() for _ in range(config["num_new_tasks"])]
                        
# tasks implied by meta mappings, network will also be trained on these  
config["implied_base_tasks"] = [] 
config["implied_new_tasks"] = []

### END PARAMATERS (finally) ##################################

def _stringify_polynomial(p):
    """Helper for printing, etc."""
    return p.to_symbols(strips_spaces=True)

def _save_config(filename, config):
    with open(filename, "w") as fout:
        fout.write("key, value\n")
        for key, value in config.items():
            fout.write(key + ", " + str(value) + "\n")

var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG')

def _get_meta_pairings(base_tasks, meta_tasks, meta_mappings, meta_binary_funcs):
    """Gets which tasks map to which other tasks under the meta_tasks (i.e. the
    part of the meta datasets which is precomputable)"""
    all_meta_tasks = meta_tasks + meta_mappings + meta_binary_funcs
    meta_pairings = {mt: [] for mt in all_meta_tasks}
    implied_tasks = []
    for mt in all_meta_tasks:
        if mt == "square":
            for poly in base_tasks: 
                other = poly ** 2
                implied_tasks.append(other)
                meta_pairings[mt].append((_stringify_polynomial(task),
                                          _stringify_polynomial(other)))
        elif mt[:3] == "add":
            c = float(mt[4:])
            for poly in base_tasks: 
                other = poly + c
                implied_tasks.append(other)
                meta_pairings[mt].append((_stringify_polynomial(task),
                                          _stringify_polynomial(other)))
        elif mt[:4] == "mult":
            c = float(mt[5:])
            for poly in base_tasks: 
                other = poly * c
                implied_tasks.append(other)
                meta_pairings[mt].append((_stringify_polynomial(task),
                                          _stringify_polynomial(other)))
        elif mt[:7] == "permute":
            perm = [int(c) for c in mt[8:]] 
            for poly in base_tasks: 
                other = poly.permute_vars(perm) 
                implied_tasks.append(other)
                meta_pairings[mt].append((_stringify_polynomial(task),
                                          _stringify_polynomial(other)))
        elif mt == "is_constant_polynomial":
            for poly in base_tasks: 
                truth_val = poly.my_max_degree == 0 
                meta_pairings[mt].append((_stringify_polynomial(task),
                                          1*truth_val))
        elif mt == "is_intercept_nonzero":
            for poly in base_tasks: 
                truth_val = "1" in poly.coefficients and poly.coefficients["1"] != 0.
                meta_pairings[mt].append((_stringify_polynomial(task),
                                          1*truth_val))
        elif mt[:3] == "is_":
            var = mt.split("_")[1]
            for poly in base_tasks: 
                truth_val = var in poly.relevant_variables 
                meta_pairings[mt].append((_stringify_polynomial(task),
                                          1*truth_val))
        elif mt[:6] == "binary":
            operation = mt[7:]
            if operation not in ["sum", "mult"]:
                raise ValueError("Unknown meta task: %s" % meta_task)
            pairings = 
            for poly1 in base_tasks:
                for poly2 in base_tasks:
                    if operation == "sum":
                        other = poly1 + poly2
                    elif operation == "mult":
                        other = poly1 * poly2
                implied_tasks.append(other)
                meta_pairings[mt].append((_stringify_polynomial(task),
                                          _stringify_polynomial(other)))

        else: 
            raise ValueError("Unknown meta task: %s" % meta_task)

    return meta_pairings, implied_tasks


class memory_buffer(object):
    """Essentially a wrapper around numpy arrays that handles inserting and
    removing."""
    def __init__(self, length, input_width, outcome_width):
        self.length = length 
        self.curr_index = 0 
        self.input_buffer = np.zeros([length, input_width])
        self.outcome_buffer = np.zeros([length, outcome_width])

    def insert(self, input_mat, outcome_mat):
        num_events = len(input_mat)
        if num_events > self.length:
            num_events = length
            self.input_buffer = input_mat[-length:, :]
            self.outcome_buffer = outcome_mat[-length:, :]
            self.curr_index = 0.
            return
        end_offset = num_events + self.curr_index
        if end_offset > self.length: 
            back_off = self.length - end_offset
            num_to_end = num_events + back_off
            self.input_buffer[:-back_off, :] = input_mat[num_to_end:, :] 
            self.outcome_buffer[:-back_off, :] = outcome_mat[num_to_end:, :] 
        else: 
            back_off = end_offset
            num_to_end = num_events
        self.input_buffer[self.curr_index:back_off, :] = input_mat[:num_to_end, :] 
        self.outcome_buffer[self.curr_index:back_off, :] = outcome_mat[:num_to_end, :] 
        self.curr_index = np.abs(back_off)

    def get_memories(self): 
        return self.input_buffer, self.outcome_buffer

