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
#    "num_hidden_language": 512,

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
#    "min_language_learning_rate": 1e-8,
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
    "poly_coeff_sd": 2.5,

    "meta_add_vals": [-3, -2, -1, 1, 3],
    "meta_mult_vals": [-3, -1, 3],
    "num_meta_binary_pairs": 200, # for binary tasks like multiplying 
                                  # polynomials, how many pairs does the 
                                  # system see?
    "new_meta_tasks": [],
    "new_meta_mappings": ["permute_3210", "add_%f" % 2., "add_%f" % -2., "mult_%f" % 2., "mult_%f" % -2.],
    
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

config["base_tasks"] = [poly_fam.sample_polynomial(coefficient_sd=config["poly_coeff_sd"]) for _ in range(config["num_base_tasks"])]
config["new_tasks"] = [poly_fam.sample_polynomial(coefficient_sd=config["poly_coeff_sd"]) for _ in range(config["num_new_tasks"])]
                        
# tasks implied by meta mappings, network will also be trained on these  
config["implied_base_tasks"] = [] 
config["implied_new_tasks"] = []

### END PARAMATERS (finally) ##################################

def _stringify_polynomial(p):
    """Helper for printing, etc."""
    return p.to_symbols(strip_spaces=True)

def _save_config(filename, config):
    with open(filename, "w") as fout:
        fout.write("key, value\n")
        for key, value in config.items():
            fout.write(key + ", " + str(value) + "\n")

var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG')


def get_distinct_random_choices(values, num_choices_per, num_sets,
                                replace=False):
    sets = []
    while len(sets) < num_sets:
        candidate_set = set(np.random.choice(values, num_choices_per, 
                                             replace=replace))
        if candidate_set not in sets:
            sets.append(candidate_set)
        
    return [np.random.permutation(list(s)) for s in sets]


def _get_meta_pairings(base_tasks, meta_tasks, meta_mappings, meta_binary_funcs):
    """Gets which tasks map to which other tasks under the meta_tasks (i.e. the
    part of the meta datasets which is precomputable)"""
    all_meta_tasks = meta_tasks + meta_mappings + meta_binary_funcs
    meta_pairings = {mt: [] for mt in all_meta_tasks}
    implied_tasks = []
    for mt in all_meta_tasks:
        if mt == "square":
            for poly in base_tasks: 
                if poly.my_max_degree**2 > poly.family.max_degree:
                    continue
                other = poly ** 2
                implied_tasks.append(other)
                meta_pairings[mt].append((_stringify_polynomial(poly),
                                          _stringify_polynomial(other)))
        elif mt[:3] == "add":
            c = float(mt[4:])
            for poly in base_tasks: 
                other = poly + c
                implied_tasks.append(other)
                meta_pairings[mt].append((_stringify_polynomial(poly),
                                          _stringify_polynomial(other)))
        elif mt[:4] == "mult":
            c = float(mt[5:])
            for poly in base_tasks: 
                other = poly * c
                implied_tasks.append(other)
                meta_pairings[mt].append((_stringify_polynomial(poly),
                                          _stringify_polynomial(other)))
        elif mt[:7] == "permute":
            perm = [int(c) for c in mt[8:]] 
            for poly in base_tasks: 
                other = poly.permute_vars(perm) 
                implied_tasks.append(other)
                meta_pairings[mt].append((_stringify_polynomial(poly),
                                          _stringify_polynomial(other)))
        elif mt == "is_constant_polynomial":
            for poly in base_tasks: 
                truth_val = poly.my_max_degree == 0 
                meta_pairings[mt].append((_stringify_polynomial(poly),
                                          1*truth_val))
        elif mt == "is_intercept_nonzero":
            for poly in base_tasks: 
                truth_val = "1" in poly.coefficients and poly.coefficients["1"] != 0.
                meta_pairings[mt].append((_stringify_polynomial(poly),
                                          1*truth_val))
        elif mt[:3] == "is_":
            var = mt.split("_")[1]
            for poly in base_tasks: 
                truth_val = var in poly.relevant_variables 
                meta_pairings[mt].append((_stringify_polynomial(poly),
                                          1*truth_val))
        elif mt[:6] == "binary":
            operation = mt[7:]
            if operation not in ["sum", "mult"]:
                raise ValueError("Unknown meta task: %s" % meta_task)
            pairings = get_distinct_random_choices(
                values=base_tasks, num_choices_per=2,
                num_sets=config["num_meta_binary_pairs"], replace=False)  
            for poly1, poly2 in pairings:
                if operation == "sum":
                    other = poly1 + poly2
                elif operation == "mult":
                    if poly1.my_max_degree + poly2.my_max_degree > poly1.family.max_degree:
                        continue
                    other = poly1 * poly2
                implied_tasks.append(other)
                meta_pairings[mt].append((_stringify_polynomial(poly1),
                                          _stringify_polynomial(poly2),
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


class meta_model(object):
    """A meta-learning model for polynomials."""
    def __init__(self, config):
        """args:
            config: a config dict, see above
        """
        self.config = config
        self.memory_buffer_size = config["memory_buffer_size"]
        self.meta_batch_size = config["meta_batch_size"]
        self.num_input = config["num_input"]
        self.num_output = config["num_output"]
        self.tkp = 1. - config["train_drop_prob"] # drop prob -> keep prob

        base_tasks = config["base_tasks"]
        base_meta_tasks = config["base_meta_tasks"]
        base_meta_mappings = config["base_meta_mappings"]
        base_meta_binary_funcs = config["base_meta_binary_funcs"]

        new_tasks = config["new_tasks"]
        new_meta_tasks = config["new_meta_tasks"]
        new_meta_mappings = config["new_meta_mappings"]

        # base datasets / memory_buffers
        self.base_tasks = base_tasks
        self.base_task_names = [_stringify_polynomial(t) for t in base_tasks]

        # new datasets / memory_buffers
        self.new_tasks = new_tasks
        self.new_task_names = [_stringify_polynomial(t) for t in new_tasks]

        self.all_base_tasks = self.base_tasks + self.new_tasks

        self.base_meta_tasks = base_meta_tasks
        self.base_meta_mappings = base_meta_mappings
        self.base_meta_binary_funcs = base_meta_binary_funcs
        self.all_base_meta_tasks = base_meta_tasks + base_meta_mappings + base_meta_binary_funcs
        self.new_meta_tasks = new_meta_tasks
        self.new_meta_mappings = new_meta_mappings
        self.all_new_meta_tasks = new_meta_tasks + new_meta_mappings
        self.all_meta_tasks = self.all_base_meta_tasks + self.all_new_meta_tasks
        self.meta_dataset_cache = {t: {} for t in self.all_meta_tasks} # will cache datasets for a certain number of epochs
                                                                       # both to speed training and to keep training targets
                                                                       # consistent

        self.all_initial_tasks = self.base_tasks + self.all_base_meta_tasks
        self.all_new_tasks = self.new_tasks + self.all_new_meta_tasks

        # think that's enough redundant variables?
        self.meta_pairings_base, self.base_tasks_implied = _get_meta_pairings(
            self.base_tasks, self.base_meta_tasks, self.base_meta_mappings,
            self.base_meta_binary_funcs)

        self.meta_pairings_full, self.full_tasks_implied = _get_meta_pairings(
            self.base_tasks + self.new_tasks,
            self.base_meta_tasks + self.new_meta_tasks,
            self.base_meta_mappings + self.new_meta_mappings,
            self.base_meta_binary_funcs)

        self.base_tasks_implied_names = [_stringify_polynomial(t) for t in self.base_tasks_implied]
        self.full_tasks_implied_names = [_stringify_polynomial(t) for t in self.full_tasks_implied]

        self.memory_buffers = {_stringify_polynomial(t): memory_buffer(
            self.memory_buffer_size, self.num_input,
            self.num_output) for t in self.all_base_tasks + self.base_tasks_implied + [x for x in self.full_tasks_implied if x not in self.base_tasks_implied]}

        self.all_tasks = self.all_initial_tasks + self.all_new_tasks + self.full_tasks_implied
        self.num_tasks = num_tasks = len(self.all_tasks)

#        for key, value in self.meta_pairings_base.items():
#            print(key)
#            print(value)
        
        # network

        # base task input
        input_size = config["num_input"]
        output_size = config["num_output"]

        self.base_input_ph = tf.placeholder(
            tf.float32, shape=[None, input_size])
        self.base_target_ph = tf.placeholder(
            tf.float32, shape=[None, output_size])

        self.lr_ph = tf.placeholder(tf.float32)
        self.keep_prob_ph = tf.placeholder(tf.float32) # dropout keep prob

        num_hidden = config["num_hidden"]
        num_hidden_hyper = config["num_hidden_hyper"]
        internal_nonlinearity = config["internal_nonlinearity"]
        output_nonlinearity = config["output_nonlinearity"]
        input_processing_1 = slim.fully_connected(self.base_input_ph, num_hidden,
                                                  activation_fn=internal_nonlinearity)

        input_processing_2 = slim.fully_connected(input_processing_1, num_hidden,
                                                  activation_fn=internal_nonlinearity)

        processed_input = slim.fully_connected(input_processing_2, num_hidden_hyper,
                                               activation_fn=internal_nonlinearity)
        self.processed_input = processed_input

        all_target_processor_nontf = random_orthogonal(num_hidden_hyper)[:, :output_size + 1]
        self.target_processor_nontf = all_target_processor_nontf[:, :output_size]
        self.target_processor = tf.get_variable('target_processor',
                                                shape=[num_hidden_hyper, output_size],
                                                initializer=tf.constant_initializer(self.target_processor_nontf))
        processed_targets = tf.matmul(self.base_target_ph, tf.transpose(self.target_processor))

        def _output_mapping(X):
            """hidden space mapped back to T/F output logits"""
            res = tf.matmul(X, self.target_processor)
            return res

        # meta task input
        self.meta_input_ph = tf.placeholder(tf.float32, shape=[None, num_hidden_hyper])
        self.meta_input_2_ph = tf.placeholder(tf.float32, shape=[None, num_hidden_hyper])
        self.meta_target_ph = tf.placeholder(tf.float32, shape=[None, num_hidden_hyper])
        self.meta_class_ph = tf.placeholder(tf.float32, shape=[None, 1])
        # last is for meta classification tasks

        self.class_processor_nontf = all_target_processor_nontf[:, output_size:]
        self.class_processor = tf.get_variable('class_processor',
                                               shape=self.class_processor_nontf.shape,
                                               initializer=tf.constant_initializer(self.class_processor_nontf))
        processed_class = tf.matmul(self.meta_class_ph, tf.transpose(self.class_processor))

        # function embedding "guessing" network / meta network
        # {(emb_in, emb_out), ...} -> emb
        self.guess_input_mask_ph = tf.placeholder(tf.bool, shape=[None]) # which datapoints get excluded from the guess

        def _meta_network(embedded_inputs, embedded_targets,
                           mask_ph=self.guess_input_mask_ph, reuse=True):
            with tf.variable_scope('meta/network', reuse=reuse):
                guess_input = tf.concat([embedded_inputs,
                                         embedded_targets], axis=-1)
                guess_input = tf.boolean_mask(guess_input,
                                              self.guess_input_mask_ph)
                guess_input = tf.nn.dropout(guess_input, self.keep_prob_ph)

                gh_1 = slim.fully_connected(guess_input, num_hidden_hyper,
                                            activation_fn=internal_nonlinearity)
                gh_1 = tf.nn.dropout(gh_1, self.keep_prob_ph)
                gh_2 = slim.fully_connected(gh_1, num_hidden_hyper,
                                            activation_fn=internal_nonlinearity)
                gh_2 = tf.nn.dropout(gh_2, self.keep_prob_ph)
                gh_2b = tf.reduce_max(gh_2, axis=0, keep_dims=True)
                gh_3 = slim.fully_connected(gh_2b, num_hidden_hyper,
                                            activation_fn=internal_nonlinearity)
                gh_3 = tf.nn.dropout(gh_3, self.keep_prob_ph)

                guess_embedding = slim.fully_connected(gh_3, num_hidden_hyper,
                                                       activation_fn=None)
                guess_embedding = tf.nn.dropout(guess_embedding, self.keep_prob_ph)
                return guess_embedding

        self.guess_base_function_emb = _meta_network(processed_input,
                                                     processed_targets,
                                                     reuse=False)

        self.guess_meta_t_function_emb = _meta_network(self.meta_input_ph,
                                                       processed_class)

        self.guess_meta_m_function_emb = _meta_network(self.meta_input_ph,
                                                       self.meta_target_ph)

        # for binary tasks 
        def _combine_inputs(inputs_1, inputs_2, reuse=True):
            with tf.variable_scope('meta/bi_combination', reuse=reuse):
                c_input = tf.concat([inputs_1,
                                     inputs_2], axis=-1)
                c_input = tf.nn.dropout(c_input, self.keep_prob_ph)

                ch_1 = slim.fully_connected(c_input, num_hidden_hyper,
                                            activation_fn=internal_nonlinearity)
                ch_1 = tf.nn.dropout(ch_1, self.keep_prob_ph)
                ch_2 = slim.fully_connected(ch_1, num_hidden_hyper,
                                            activation_fn=internal_nonlinearity)
                ch_2 = tf.nn.dropout(ch_2, self.keep_prob_ph)
                ch_3 = slim.fully_connected(ch_2, num_hidden_hyper,
                                            activation_fn=internal_nonlinearity)
                combined = tf.nn.dropout(ch_3, self.keep_prob_ph)
                # also include direct inputs averaged
                self.combination_weight = tf.get_variable('combination_weight',
                                                          shape=[],
                                                          initializer=tf.constant_initializer(0.))
                c_w = tf.nn.sigmoid(self.combination_weight) 
                combined = (c_w) * combined + (1.- c_w) * 0.5 * (inputs_1 + inputs_2) 
                return combined 

        self.combined_meta_inputs = _combine_inputs(self.meta_input_ph,
                                                    self.meta_input_2_ph,
                                                    reuse=False)

        self.guess_meta_bf_function_emb = _meta_network(
            self.combined_meta_inputs, self.meta_target_ph)
    
#        print(self.combined_meta_inputs)
#        print(self.guess_base_function_emb)
#        print(self.guess_meta_bf_function_emb)

        # hyper_network: emb -> (f: emb -> emb)
        self.feed_embedding_ph = tf.placeholder(np.float32,
                                                [1, num_hidden_hyper])

        num_task_hidden_layers = config["num_task_hidden_layers"]

        tw_range = config["task_weight_weight_mult"]/np.sqrt(
            num_hidden * num_hidden_hyper) # yields a very very roughly O(1) map
        task_weight_gen_init = tf.random_uniform_initializer(-tw_range,
                                                             tw_range)

        def _hyper_network(function_embedding, reuse=True):
            with tf.variable_scope('hyper', reuse=reuse):
                hyper_hidden = function_embedding
                for _ in range(config["num_hyper_hidden_layers"]):
                    hyper_hidden = slim.fully_connected(hyper_hidden, num_hidden_hyper,
                                                        activation_fn=internal_nonlinearity)
                    hyper_hidden = tf.nn.dropout(hyper_hidden, self.keep_prob_ph)

                hidden_weights = []
                hidden_biases = []

                task_weights = slim.fully_connected(hyper_hidden, num_hidden*(num_hidden_hyper +(num_task_hidden_layers-1)*num_hidden + num_hidden_hyper),
                                                    activation_fn=None,
                                                    weights_initializer=task_weight_gen_init)
                task_weights = tf.nn.dropout(task_weights, self.keep_prob_ph)

                task_weights = tf.reshape(task_weights, [-1, num_hidden, (num_hidden_hyper + (num_task_hidden_layers-1)*num_hidden + num_hidden_hyper)])
                task_biases = slim.fully_connected(hyper_hidden, num_task_hidden_layers * num_hidden + num_hidden_hyper,
                                                   activation_fn=None)

                Wi = tf.transpose(task_weights[:, :, :num_hidden_hyper], perm=[0, 2, 1])
                bi = task_biases[:, :num_hidden]
                hidden_weights.append(Wi)
                hidden_biases.append(bi)
                for i in range(1, num_task_hidden_layers):
                    Wi = tf.transpose(task_weights[:, :, input_size+(i-1)*num_hidden:input_size+i*num_hidden], perm=[0, 2, 1])
                    bi = task_biases[:, num_hidden*i:num_hidden*(i+1)]
                    hidden_weights.append(Wi)
                    hidden_biases.append(bi)
                Wfinal = task_weights[:, :, -num_hidden_hyper:]
                bfinal = task_biases[:, -num_hidden_hyper:]

                for i in range(num_task_hidden_layers):
                    hidden_weights[i] = tf.squeeze(hidden_weights[i], axis=0)
                    hidden_biases[i] = tf.squeeze(hidden_biases[i], axis=0)

                Wfinal = tf.squeeze(Wfinal, axis=0)
                bfinal = tf.squeeze(bfinal, axis=0)
                hidden_weights.append(Wfinal)
                hidden_biases.append(bfinal)
                return hidden_weights, hidden_biases

        self.base_task_params = _hyper_network(self.guess_base_function_emb,
                                               reuse=False)
        self.meta_t_task_params = _hyper_network(self.guess_meta_t_function_emb)
        self.meta_m_task_params = _hyper_network(self.guess_meta_m_function_emb)
        self.meta_bf_task_params = _hyper_network(self.guess_meta_bf_function_emb)
        self.fed_emb_task_params = _hyper_network(self.feed_embedding_ph)

        # task network
        def _task_network(task_params, processed_input):
            hweights, hbiases = task_params
            task_hidden = processed_input
            for i in range(num_task_hidden_layers):
                task_hidden = internal_nonlinearity(
                    tf.matmul(task_hidden, hweights[i]) + hbiases[i])

            raw_output = tf.matmul(task_hidden, hweights[-1]) + hbiases[-1]

            return raw_output

        self.base_raw_output = _task_network(self.base_task_params,
                                             processed_input)
        self.base_output = _output_mapping(self.base_raw_output)

        self.base_raw_output_fed_emb = _task_network(self.fed_emb_task_params,
                                                     processed_input)
        self.base_output_fed_emb = _output_mapping(self.base_raw_output_fed_emb)

        self.meta_t_raw_output = _task_network(self.meta_t_task_params,
                                               self.meta_input_ph)
        self.meta_t_output = tf.nn.sigmoid(self.meta_t_raw_output)

        self.meta_m_output = _task_network(self.meta_m_task_params,
                                           self.meta_input_ph)

        self.meta_bf_output = _task_network(self.meta_bf_task_params,
                                            self.combined_meta_inputs)

        self.base_loss = tf.square(self.base_output - self.base_target_ph)
        self.total_base_loss = tf.reduce_mean(self.base_loss)

        self.base_fed_emb_loss = tf.square(
            self.base_output_fed_emb - self.base_target_ph)
        self.total_base_fed_emb_loss = tf.reduce_mean(self.base_fed_emb_loss)

        self.meta_t_loss = tf.reduce_sum(
            tf.square(self.meta_t_output - processed_class), axis=1)
        self.total_meta_t_loss = tf.reduce_mean(self.meta_t_loss)

        self.meta_m_loss = tf.reduce_sum(
            tf.square(self.meta_m_output - self.meta_target_ph), axis=1)
        self.total_meta_m_loss = tf.reduce_mean(self.meta_m_loss)

        self.meta_bf_loss = tf.reduce_sum(
            tf.square(self.meta_bf_output - self.meta_target_ph), axis=1)
        self.total_meta_bf_loss = tf.reduce_mean(self.meta_bf_loss)

        optimizer = tf.train.RMSPropOptimizer(self.lr_ph)

        self.base_train = optimizer.minimize(self.total_base_loss)
        self.meta_t_train = optimizer.minimize(self.total_meta_t_loss)
        self.meta_m_train = optimizer.minimize(self.total_meta_m_loss)
        self.meta_bf_train = optimizer.minimize(self.total_meta_bf_loss)

        # Saver
        self.saver = tf.train.Saver()

        # initialize
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())
        self.fill_buffers(num_data_points=config["memory_buffer_size"],
                          include_new=True)

        self.refresh_meta_dataset_cache()


    def fill_buffers(self, num_data_points=1, include_new=False):
        """Add new "experiences" to memory buffers."""
        if include_new:
            this_tasks = self.all_base_tasks
        else:
            this_tasks = self.base_tasks
        for t in this_tasks:
            buff = self.memory_buffers[_stringify_polynomial(t)]
            x_data = np.zeros([num_data_points, self.config["num_input"]])
            y_data = np.zeros([num_data_points, self.config["num_output"]])
            for point_i in range(num_data_points):
                point = t.family.sample_point()
                x_data[point_i, :] = point
                y_data[point_i, :] = t.evaluate(point)
            buff.insert(x_data, y_data)


    def _random_guess_mask(self, dataset_length, meta_batch_size=None):
        if meta_batch_size is None:
            meta_batch_size = config["meta_batch_size"]
        mask = np.zeros(dataset_length, dtype=np.bool)
        indices = np.random.permutation(dataset_length)[:meta_batch_size]
        mask[indices] = True
        return mask


    def base_train_step(self, memory_buffer, lr):
        input_buff, output_buff = memory_buffer.get_memories()
        feed_dict = {
            self.base_input_ph: input_buff,
            self.guess_input_mask_ph: self._random_guess_mask(self.memory_buffer_size),
            self.base_target_ph: output_buff,
            self.keep_prob_ph: self.tkp,
            self.lr_ph: lr
        }
        self.sess.run(self.base_train, feed_dict=feed_dict)


    def base_eval(self, memory_buffer, meta_batch_size=None):
        input_buff, output_buff = memory_buffer.get_memories()
        feed_dict = {
            self.base_input_ph: input_buff,
            self.guess_input_mask_ph: self._random_guess_mask(
                self.memory_buffer_size, meta_batch_size=meta_batch_size),
            self.base_target_ph: output_buff,
            self.keep_prob_ph: 1.
        }
        fetches = [self.total_base_loss]
        res = self.sess.run(fetches, feed_dict=feed_dict)
        return res


    def run_base_eval(self, include_new=False, sweep_meta_batch_sizes=False):
        """sweep_meta_batch_sizes: False or a list of meta batch sizes to try"""
        if include_new:
            tasks = self.all_base_tasks
        else:
            tasks = self.base_tasks

        losses = [] 
        if sweep_meta_batch_sizes:
            for meta_batch_size in sweep_meta_batch_sizes:
                this_losses = [] 
                for task in tasks:
                    task_str = _stringify_polynomial(task)
                    memory_buffer = self.memory_buffers[task_str]
                    res = self.base_eval(memory_buffer, 
                                         meta_batch_size=meta_batch_size)
                    this_losses.append(res[0])
                losses.append(this_losses)
        else:
            for task in tasks:
                task_str = _stringify_polynomial(task)
                memory_buffer = self.memory_buffers[task_str]
                res = self.base_eval(memory_buffer)
                losses.append(res[0])

        names = [_stringify_polynomial(t) for t in tasks]
        return names, losses


    def base_embedding_eval(self, embedding, memory_buffer):
        input_buff, output_buff = memory_buffer.get_memories()
        feed_dict = {
            self.keep_prob_ph: 1.,
            self.feed_embedding_ph: embedding,
            self.base_input_ph: input_buff,
            self.base_target_ph: output_buff
        }
        fetches = [self.total_base_fed_emb_loss]
        res = self.sess.run(fetches, feed_dict=feed_dict)
        return res

    
    def get_base_embedding(self, memory_buffer):
        input_buff, output_buff = memory_buffer.get_memories()
        feed_dict = {
            self.keep_prob_ph: 1.,
            self.base_input_ph: input_buff,
            self.guess_input_mask_ph: np.ones([self.memory_buffer_size]),
            self.base_target_ph: output_buff
        }
        res = self.sess.run(self.guess_base_function_emb, feed_dict=feed_dict)
        return res


    def get_combined_embedding(self, t1_embedding, t2_embedding):
        feed_dict = {
            self.keep_prob_ph: 1.,
            self.meta_input_ph: t1_embedding,
            self.meta_input_2_ph: t2_embedding
        }
        res = self.sess.run(self.combined_meta_inputs, feed_dict=feed_dict)
        return res


    def get_meta_dataset(self, meta_task, include_new=False):
        x_data = []
        x2_data = []
        y_data = []
        if include_new:
            this_base_tasks = self.meta_pairings_full[meta_task]
        else:
            this_base_tasks = self.meta_pairings_base[meta_task]
        for this_tuple in this_base_tasks:
            if len(this_tuple) == 3: # binary_func 
                task, task2, other = this_tuple
                task2_buffer = self.memory_buffers[task2]
                x2_data.append(self.get_base_embedding(task2_buffer)[0, :])
            else:
                task, other = this_tuple
            task_buffer = self.memory_buffers[task]
            x_data.append(self.get_base_embedding(task_buffer)[0, :])
            if other in [0, 1]:  # for classification meta tasks
                y_data.append([other])
            else:
                other_buffer = self.memory_buffers[other]
                y_data.append(self.get_base_embedding(other_buffer)[0, :])
        if x2_data != []: # binary func
            return {"x1": np.array(x_data), "x2": np.array(x2_data),
                    "y": np.array(y_data)}
        else:
            return {"x": np.array(x_data), "y": np.array(y_data)}


    def refresh_meta_dataset_cache(self, include_new=False):
        meta_tasks = self.all_base_meta_tasks 
        if include_new:
            meta_tasks += self.all_new_meta_tasks 

        for t in meta_tasks:
            self.meta_dataset_cache[t] = self.get_meta_dataset(t, include_new)


    def meta_loss_eval(self, meta_dataset):
        feed_dict = {
            self.keep_prob_ph: 1.,
        }
        y_data = meta_dataset["y"]
        if y_data.shape[-1] == 1:
            feed_dict[self.meta_class_ph] = y_data 
            fetch = self.total_meta_t_loss
        else:
            feed_dict[self.meta_target_ph] = y_data 
            fetch = self.total_meta_m_loss

        if "x1" in meta_dataset: 
            feed_dict[self.meta_input_ph] = meta_dataset["x1"]
            feed_dict[self.meta_input_2_ph] = meta_dataset["x2"]
            feed_dict[self.guess_input_mask_ph] =  np.ones([len(meta_dataset["x1"])])
            fetch = self.total_meta_bf_loss
        else:
            feed_dict[self.meta_input_ph] = meta_dataset["x"]
            feed_dict[self.guess_input_mask_ph] =  np.ones([len(meta_dataset["x"])])

        return self.sess.run(fetch, feed_dict=feed_dict)
        

    def run_meta_loss_eval(self, include_new=False):
        meta_tasks = self.all_base_meta_tasks 
        if include_new:
            meta_tasks += self.new_meta_tasks 

        names = []
        losses = []
        for t in meta_tasks:
            meta_dataset = self.meta_dataset_cache[t]
            loss = self.meta_loss_eval(meta_dataset)
            names.append(t)
            losses.append(loss)

        return names, losses


    def get_meta_embedding(self, meta_dataset):
        feed_dict = {
            self.keep_prob_ph: 1.,
            self.guess_input_mask_ph: np.ones([len(meta_dataset["x"])])
        }
        y_data = meta_dataset["y"]
        if y_data.shape[-1] == 1:
            feed_dict[self.meta_class_ph] = y_data 
            fetch = self.guess_meta_t_function_emb
        else:
            feed_dict[self.meta_target_ph] = y_data 
            fetch = self.guess_meta_m_function_emb

        if "x1" in meta_dataset: 
            feed_dict[self.meta_input_ph] = meta_dataset["x1"]
            feed_dict[self.meta_input_2_ph] = meta_dataset["x2"]
            fetch = self.guess_meta_bf_function_emb
        else:
            feed_dict[self.meta_input_ph] = meta_dataset["x"]

        return self.sess.run(fetch, feed_dict=feed_dict)


    def get_meta_outputs(self, meta_dataset, new_dataset=None):
        """Get new dataset mapped according to meta_dataset, or just outputs
        for original dataset if new_dataset is None"""
        meta_class = meta_dataset["y"].shape[-1] == 1

        if new_dataset is not None:
            if "x" in meta_dataset:
                this_x = np.concatenate([meta_dataset["x"], new_dataset["x"]], axis=0)
                new_x = new_dataset["x"] 
                this_mask = np.zeros(len(this_x), dtype=np.bool)
                this_mask[:len(meta_dataset["x"])] = True # use only these to guess
            else:
                this_x1 = np.concatenate([meta_dataset["x1"], new_dataset["x1"]], axis=0)
                this_x2 = np.concatenate([meta_dataset["x2"], new_dataset["x2"]], axis=0)
                new_x = new_dataset["x1"] # for size only 
                this_mask = np.zeros(len(this_x1), dtype=np.bool)
                this_mask[:len(meta_dataset["x1"])] = True # use only these to guess

            if meta_class:
                this_y = np.concatenate([meta_dataset["y"], np.zeros([len(new_x)])], axis=0)
            else:
                this_y = np.concatenate([meta_dataset["y"], np.zeros_like(new_x)], axis=0)

        else:
            if "x" in meta_dataset:
                this_x = meta_dataset["x"]
                this_mask = np.ones(len(this_x), dtype=np.bool)
            else:
                this_x1 = meta_dataset["x1"]
                this_x2 = meta_dataset["x2"]
                this_mask = np.ones(len(this_x1), dtype=np.bool)
            this_y = meta_dataset["y"]

        feed_dict = {
            self.keep_prob_ph: 1.,
            self.guess_input_mask_ph: this_mask 
        }
        if meta_class:
            feed_dict[self.meta_class_ph] = this_y 
            this_fetch = self.meta_t_output 
        else:
            feed_dict[self.meta_target_ph] = this_y
            this_fetch = self.meta_m_output 

        if "x1" in meta_dataset: 
            feed_dict[self.meta_input_ph] = this_x1 
            feed_dict[self.meta_input_2_ph] = this_x2 
            fetch = self.meta_bf_output
        else:
            feed_dict[self.meta_input_ph] = this_x 

        res = self.sess.run(this_fetch, feed_dict=feed_dict)

        if new_dataset is not None:
            return res[len(meta_dataset["y"]):, :]
        else:
            return res


    def run_meta_true_eval(self, include_new=False):
        """Evaluates true meta loss, i.e. the accuracy of the model produced
           by the embedding output by the meta task"""
        if include_new:
            meta_tasks = self.base_meta_mappings + self.new_meta_mappings
            meta_pairings = self.meta_pairings_full
        else:
            meta_tasks = self.base_meta_mappings
            meta_pairings = self.meta_pairings_base
        meta_binary_funcs = self.base_meta_binary_funcs

        names = []
        losses = []
        for meta_task in meta_tasks:
            meta_dataset = self.meta_dataset_cache[meta_task]
            if meta_dataset == {}: # new tasks aren't cached
                meta_dataset = self.get_meta_dataset(meta_task, include_new) 

            for task, other in meta_pairings[meta_task]:
                task_buffer = self.memory_buffers[task]
                task_embedding = self.get_base_embedding(task_buffer)

                other_buffer = self.memory_buffers[other]

                mapped_embedding = self.get_meta_outputs(
                    meta_dataset, {"x": task_embedding})

                names.append(meta_task + ":" + task + "->" + other)
                this_loss = self.base_embedding_eval(mapped_embedding, other_buffer)[0]
                losses.append(this_loss)
        for meta_task in meta_binary_funcs:
            meta_dataset = self.meta_dataset_cache[meta_task]
            for task1, task2, other in meta_pairings[meta_task]:
                task1_buffer = self.memory_buffers[task1]
                task1_embedding = self.get_base_embedding(task1_buffer)
                task2_buffer = self.memory_buffers[task2]
                task2_embedding = self.get_base_embedding(task2_buffer)

                other_buffer = self.memory_buffers[other]

                mapped_embedding = self.get_meta_outputs(
                    meta_dataset, {"x1": task1_embedding,
                                   "x2": task2_embedding})

                names.append(meta_task + ":" + task + "->" + other)
                this_loss = self.base_embedding_eval(mapped_embedding, other_buffer)[0]
                losses.append(this_loss)

        return names, losses 


    def meta_train_step(self, meta_dataset, meta_lr):
        y_data = meta_dataset["y"]
        if "x" in meta_dataset:
            feed_dict = {
                self.keep_prob_ph: self.tkp,
                self.meta_input_ph: meta_dataset["x"],
                self.guess_input_mask_ph: np.ones([len(meta_dataset["x"])]),
                self.lr_ph: meta_lr
            }
            meta_class = y_data.shape[-1] == 1
            if meta_class:
                feed_dict[self.meta_class_ph] = y_data
                op = self.meta_t_train
            else:
                feed_dict[self.meta_target_ph] = y_data
                op = self.meta_m_train
        else:
            feed_dict = {
                self.keep_prob_ph: self.tkp,
                self.meta_input_ph: meta_dataset["x1"],
                self.meta_input_2_ph: meta_dataset["x2"],
                self.guess_input_mask_ph: np.ones([len(meta_dataset["x1"])]),
                self.lr_ph: meta_lr
            }
            feed_dict[self.meta_target_ph] = y_data
            op = self.meta_bf_train

        self.sess.run(op, feed_dict=feed_dict)


    def save_parameters(self, filename):
        self.saver.save(self.sess, filename)


    def run_training(self, filename_prefix, num_epochs, include_new=False):
        """Train model on base and meta tasks, if include_new include also
        the new ones."""
        config = self.config
        loss_filename = filename_prefix + "_losses.csv"
        sweep_filename = filename_prefix + "_sweep_losses.csv"
        meta_filename = filename_prefix + "_meta_true_losses.csv"
        with open(loss_filename, "w") as fout, open(meta_filename, "w") as fout_meta:
            base_names, base_losses = self.run_base_eval(
                include_new=include_new)
            meta_names, meta_losses = self.run_meta_loss_eval(
                include_new=include_new)
            meta_true_names, meta_true_losses = self.run_meta_true_eval(
                include_new=include_new)

            fout.write("epoch, " + ", ".join(base_names + meta_names) + "\n")
            fout_meta.write("epoch, " + ", ".join(meta_true_names) + "\n")

            base_loss_format = ", ".join(["%f" for _ in base_names]) + "\n"
            loss_format = ", ".join(["%f" for _ in base_names + meta_names]) + "\n"
            meta_true_format = ", ".join(["%f" for _ in meta_true_names]) + "\n"
            
            s_epoch  = "0, "
            curr_losses = s_epoch + (loss_format % tuple(
                base_losses + meta_losses))
            curr_meta_true = s_epoch + (meta_true_format % tuple(
                meta_true_losses))
            fout.write(curr_losses)
            fout_meta.write(curr_meta_true)

            if config["sweep_meta_batch_sizes"] is not None:
                with open(sweep_filename, "w") as fout_sweep:
                    sweep_names, sweep_losses = self.run_base_eval(
                        include_new=include_new, sweep_meta_batch_sizes=config["sweep_meta_batch_sizes"])
                    fout_sweep.write("epoch, size, " + ", ".join(base_names) + "\n")
                    for i, swept_batch_size in enumerate(config["sweep_meta_batch_sizes"]):
                        swept_losses = s_epoch + ("%i, " % swept_batch_size) + (base_loss_format % tuple(sweep_losses[i]))
                        fout_sweep.write(swept_losses)

            if include_new:
                tasks = self.all_tasks
                learning_rate = config["new_init_learning_rate"]
                meta_learning_rate = config["new_init_meta_learning_rate"]
            else:
                tasks = self.all_initial_tasks
                learning_rate = config["init_learning_rate"]
                meta_learning_rate = config["init_meta_learning_rate"]

            save_every = config["save_every"]
            early_stopping_thresh = config["early_stopping_thresh"]
            lr_decays_every = config["lr_decays_every"]
            lr_decay = config["lr_decay"]
            meta_lr_decay = config["meta_lr_decay"]
            min_learning_rate = config["min_learning_rate"]
            min_meta_learning_rate = config["min_meta_learning_rate"]
            for epoch in range(1, num_epochs+1):
                if epoch % config["refresh_mem_buffs_every"] == 0:
                    self.fill_buffers(num_data_points=config["memory_buffer_size"],
                                      include_new=True)
                if epoch % config["refresh_meta_cache_every"] == 0:
                    self.refresh_meta_dataset_cache(include_new=include_new)

                order = np.random.permutation(len(tasks))
                for task_i in order:
                    task = tasks[task_i]
                    if isinstance(task, str):
                        dataset = self.meta_dataset_cache[task]
                        self.meta_train_step(dataset, meta_learning_rate)
                    else:
                        str_task = _stringify_polynomial(task)
                        memory_buffer = self.memory_buffers[str_task]
                        self.base_train_step(memory_buffer, learning_rate)

                if epoch % save_every == 0:
                    s_epoch  = "%i, " % epoch
                    _, base_losses = self.run_base_eval(
                        include_new=include_new)
                    _, meta_losses = self.run_meta_loss_eval(
                        include_new=include_new)
                    _, meta_true_losses = self.run_meta_true_eval(
                        include_new=include_new)
                    curr_losses = s_epoch + (loss_format % tuple(
                        base_losses + meta_losses))
                    curr_meta_true = s_epoch + (meta_true_format % tuple(meta_true_losses))
                    fout.write(curr_losses)
                    fout_meta.write(curr_meta_true)
                    print(curr_losses)
                    if np.all(curr_losses < early_stopping_thresh):
                        print("Early stop!")
                        break


                if epoch % lr_decays_every == 0 and epoch > 0:
                    if learning_rate > min_learning_rate:
                        learning_rate *= lr_decay

                    if meta_learning_rate > min_meta_learning_rate:
                        meta_learning_rate *= meta_lr_decay

            if config["sweep_meta_batch_sizes"] is not None:
                with open(sweep_filename, "a") as fout_sweep:
                    sweep_names, sweep_losses = self.run_base_eval(
                        include_new=include_new, sweep_meta_batch_sizes=config["sweep_meta_batch_sizes"])
                    for i, swept_batch_size in enumerate(config["sweep_meta_batch_sizes"]):
                        swept_losses = s_epoch + ("%i, " % swept_batch_size) + (base_loss_format % tuple(sweep_losses[i]))
                        fout_sweep.write(swept_losses)


## running stuff

for run_i in range(config["run_offset"], config["run_offset"]+config["num_runs"]):
    np.random.seed(run_i)
    tf.set_random_seed(run_i)
    filename_prefix = config["output_dir"] + "run%i" % run_i
    print("Now running %s" % filename_prefix)
    _save_config(filename_prefix + "_config.csv", config)


    model = meta_model(config)
#    model.save_embeddings(filename=filename_prefix + "_init_embeddings.csv",
#                          include_new=False)
    model.run_training(filename_prefix=filename_prefix,
                       num_epochs=config["max_base_epochs"],
                       include_new=False)
    model.save_parameters(filename_prefix + "_guess_checkpoint")
#    model.save_embeddings(filename=filename_prefix + "_guess_embeddings.csv",
#                          include_new=True)

    model.run_training(filename_prefix=filename_prefix + "_new",
                       num_epochs=config["max_new_epochs"],
                       include_new=True)
    model.save_parameters(filename_prefix + "_final_checkpoint")

#    model.save_embeddings(filename=filename_prefix + "_final_embeddings.csv",
#                          include_new=True)

    tf.reset_default_graph()
