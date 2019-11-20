from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from copy import deepcopy

import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import warnings

from .configs import default_architecture_config
from .etc.orthogonal_matrices import random_orthogonal


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


# default IO nets are those used for the original polynomials experiments
def default_input_processor(input_ph, IO_num_hidden, z_dim,
                            internal_nonlinearity, reuse=False):
    with tf.variable_scope('input_processor', reuse=reuse):
        input_processing_1 = slim.fully_connected(input_ph, IO_num_hidden,
                                                  activation_fn=internal_nonlinearity)

        input_processing_2 = slim.fully_connected(input_processing_1, IO_num_hidden,
                                                  activation_fn=internal_nonlinearity)

        processed_input = slim.fully_connected(input_processing_2, z_dim,
                                               activation_fn=None)
    return processed_input

def default_target_processor(targets, IO_num_hidden, z_dim,
                             internal_nonlinearity, reuse=False):
    with tf.variable_scope('target_processor', reuse=reuse):
        output_size = targets.get_shape()[-1]
        target_processor_nontf = random_orthogonal(z_dim)[:, :output_size + 1]
        target_processor = tf.get_variable('target_processor',
                                            shape=[z_dim, output_size],
                                            initializer=tf.constant_initializer(target_processor_nontf[:, :-1]))
        meta_class_processor = tf.get_variable('meta_class_processor',
                                            shape=[z_dim, 1],
                                            initializer=tf.constant_initializer(target_processor_nontf[:, -1:]))
        
        processed_targets = tf.matmul(targets, tf.transpose(target_processor))
    return processed_targets, target_processor, meta_class_processor

def default_outcome_processor(outcomes, IO_num_hidden, z_dim,
                              internal_nonlinearity, reuse=False):
    with tf.variable_scope('outcome_processor', reuse=reuse):
        outcome_processing_1 = slim.fully_connected(
            outcomes, IO_num_hidden, activation_fn=internal_nonlinearity)
        outcome_processing_2 = slim.fully_connected(
            outcome_processing_1, IO_num_hidden, activation_fn=internal_nonlinearity)
        res = slim.fully_connected(outcome_processing_2, z_dim,
                                    activation_fn=internal_nonlinearity)
    return res

def default_meta_class_processor(targets, meta_class_processor_var):
    processed_targets = tf.matmul(targets, tf.transpose(meta_class_processor_var))
    return processed_targets


def default_output_processor(output_embeddings, target_processor):
    processed_outputs = tf.matmul(output_embeddings, target_processor)
    return processed_outputs 


def default_base_loss(outputs, targets):
    return tf.reduce_mean(tf.square(outputs - targets))


def default_meta_loss(outputs, targets):
    batch_losses = tf.reduce_sum(tf.square(outputs - targets), axis=1)
    return tf.reduce_mean(batch_losses)


def default_meta_class_loss(logits, targets):
    batch_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=targets, logits=logits)
    return tf.reduce_mean(batch_losses)


def _pad(l, length, pad_token="PAD"):
    return [pad_token]*(length - len(l)) + l


def save_config(filename, config):
    with open(filename, "w") as fout:
        fout.write("key, value\n")
        for key, value in config.items():
            fout.write(key + ", " + str(value) + "\n")


class HoMM_model(object):
    """A base Homoiconic Meta-mapping model."""
    def __init__(self, run_config, architecture_config=None,
                 input_processor=None, target_processor=None,
                 output_processor=None, outcome_processor=None,
                 meta_class_processor=None, base_loss=None,
                 meta_loss=None, meta_class_loss=None):
        """Set up data structures, build the network.
        
        The child classes should call this with a super call, overriding the
        _pre_ and _post_build_calls methods as necessary. In particular, the
        _pre_build_calls function should be used to add any
        """
        if architecture_config is None:
            architecture_config = default_architecture_config.default_architecture_config 
        else:
            architecture_config = architecture_config
        self.run_config = deepcopy(run_config)
        self.architecture_config = deepcopy(architecture_config)
        self.filename_prefix = "%srun%i_" %(self.run_config["output_dir"],
                                            self.run_config["this_run"])


        # set up filenames
        self.run_config["run_config_filename"] = self.filename_prefix + "run_config.csv"
        self.run_config["architecture_config_filename"] = self.filename_prefix + "architecture_config.csv"
        self.run_config["loss_filename"] = self.filename_prefix + "losses.csv"
        self.run_config["meta_filename"] = self.filename_prefix + "meta_true_losses.csv"
        self.run_config["lang_filename"] = self.filename_prefix + "language_losses.csv"
        if not os.path.exists(self.run_config["output_dir"]):
            os.makedirs(self.run_config["output_dir"])

        # data structures to be
        self.num_tasks = 0
        self.base_train_tasks = [] 
        self.base_eval_tasks = [] 
        self.meta_class_train_tasks = []
        self.meta_class_eval_tasks = []
        self.meta_map_train_tasks = []
        self.meta_map_eval_tasks = []

        self.meta_pairings = {}

        # setup functions
        self._pre_build_calls()  # tasks should be added to lists here
        self._task_processing()

        # network and init
        self._build_architecture(
            architecture_config=architecture_config, 
            input_processor=input_processor,
            target_processor=target_processor,
            output_processor=output_processor,
            outcome_processor=outcome_processor,
            meta_class_processor=meta_class_processor,
            base_loss=base_loss,
            meta_loss=meta_loss,
            meta_class_loss=meta_class_loss)
        self._post_build_calls()
        self._sess_and_init()


    def _pre_build_calls(self):
        """Should be overridden to do something before building the net."""
        raise NotImplementedError("You must override the _pre_build_calls "
                                  "method to add your base and meta tasks.")


    def _task_processing(self):
        """Create the memory buffers, etc. for the tasks."""
        
        # update config
        self.run_config["base_train_tasks"] = self.base_train_tasks
        self.run_config["base_eval_tasks"] = self.base_eval_tasks
        self.run_config["meta_class_train_tasks"] = self.meta_class_train_tasks
        self.run_config["meta_class_eval_tasks"] = self.meta_class_eval_tasks
        self.run_config["meta_map_train_tasks"] = self.meta_map_train_tasks
        self.run_config["meta_map_eval_tasks"] = self.meta_map_eval_tasks

        self.task_indices = {}

        # create buffers for base tasks
        self.memory_buffers = {}
        for task in self.base_train_tasks + self.base_eval_tasks:
            self.base_task_lookup(task)

        # create structures for meta tasks
        self.meta_datasets = {}
        all_meta_tasks = self.meta_class_train_tasks + self.meta_class_eval_tasks + self.meta_map_train_tasks + self.meta_map_eval_tasks
        for mt in all_meta_tasks:
            self.meta_task_lookup(mt)

        # populate meta-task structures
        for mts, meta_class in zip([self.meta_class_train_tasks + self.meta_class_eval_tasks,
                                    self.meta_map_train_tasks + self.meta_map_eval_tasks],
                                   [True, False]):
            out_dtype = np.float32 if meta_class else np.int32
            for mt in mts:
                self.meta_datasets[mt] = {} 

                train_pairings = self.meta_pairings[mt]["train"]
                num_train = len(train_pairings)
                eval_pairings = self.meta_pairings[mt]["eval"]
                num_eval = len(eval_pairings)

                self.meta_datasets[mt]["train"] = {}
                self.meta_datasets[mt]["eval"] = {}
                self.meta_datasets[mt]["train"]["in"] = np.zeros(
                    [num_train, ], dtype=np.int32)

                self.meta_datasets[mt]["eval"]["in"] = np.zeros(
                    [num_train + num_eval], dtype=np.int32)
                eval_guess_mask = np.concatenate([np.ones([num_train],
                                                          dtype=np.bool),
                                                  np.zeros([num_eval],
                                                           dtype=np.bool)])
                self.meta_datasets[mt]["eval"]["gm"] = eval_guess_mask

                if meta_class:
                    self.meta_datasets[mt]["train"]["out"] = np.zeros(
                        [num_train, 1], dtype=out_dtype)
                    self.meta_datasets[mt]["eval"]["out"] = np.zeros(
                        [num_train + num_eval, 1], dtype=out_dtype)

                else:
                    self.meta_datasets[mt]["train"]["out"] = np.zeros(
                        [num_train, ], dtype=out_dtype)
                    self.meta_datasets[mt]["eval"]["out"] = np.zeros(
                        [num_train + num_eval], dtype=out_dtype)

                for i, (e, res) in enumerate(train_pairings):
                    e_index = self.task_indices[e]
                    self.meta_datasets[mt]["train"]["in"][i] = e_index
                    if meta_class:
                        self.meta_datasets[mt]["train"]["out"][i, 0] = res
                    else:
                        res_index = self.task_indices[res]
                        self.meta_datasets[mt]["train"]["out"][i] = res_index

                self.meta_datasets[mt]["eval"]["in"][:num_train] = self.meta_datasets[mt]["train"]["in"]
                self.meta_datasets[mt]["eval"]["out"][:num_train] = self.meta_datasets[mt]["train"]["out"]

                for i, (e, res) in enumerate(eval_pairings):
                    e_index = self.task_indices[e]
                    self.meta_datasets[mt]["eval"]["in"][num_train + i] = e_index
                    if meta_class:
                        self.meta_datasets[mt]["eval"]["out"][num_train + i, 0] = res
                    else:
                        res_index = self.task_indices[res]
                        self.meta_datasets[mt]["eval"]["out"][num_train + i] = res_index

    def _post_build_calls(self):
        """Can be overridden to do something after building, before session."""
        pass

    def _build_architecture(self, architecture_config, input_processor,
                            target_processor, output_processor,
                            outcome_processor, meta_class_processor, base_loss, 
                            meta_loss, meta_class_loss):
        self.architecture_config = architecture_config
        self.memory_buffer_size = architecture_config["memory_buffer_size"]
        self.meta_batch_size = architecture_config["meta_batch_size"]
        self.tkp = 1. - architecture_config["train_drop_prob"] # drop prob -> keep prob
        self.separate_targ_net = architecture_config["separate_target_network"]

        if self.run_config["train_language"]:
            self.vocab = architecture_config["vocab"]
            self.vocab_size = len(self.vocab)

        # network
        var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG')

        # base task input (polynomials, need to refactor)
        input_shape = architecture_config["input_shape"]
        output_shape = architecture_config["output_shape"]
        outcome_shape = architecture_config["outcome_shape"]

        self.base_input_ph = tf.placeholder(
            tf.float32, shape=[None] + input_shape)
        self.base_target_ph = tf.placeholder(
            tf.float32, shape=[None] + output_shape)

        if outcome_shape is not None: # if outcomes seen by M != targets
            self.base_outcome_ph = tf.placeholder(
                tf.float32, shape=[None] + outcome_shape)

        self.lr_ph = tf.placeholder(tf.float32)
        self.keep_prob_ph = tf.placeholder(tf.float32) # dropout keep prob

        F_num_hidden = architecture_config["F_num_hidden"]
        IO_num_hidden = architecture_config["IO_num_hidden"]
        num_hidden_hyper = architecture_config["H_num_hidden"]
        z_dim = architecture_config["z_dim"]
        internal_nonlinearity = architecture_config["internal_nonlinearity"]

        if input_processor is None:
            input_processor = lambda x: default_input_processor(
                x, IO_num_hidden, z_dim, internal_nonlinearity)
        self.processed_input = input_processor(self.base_input_ph)
        if self.separate_targ_net:
            with tf.variable_scope("target_net", reuse=False):
                self.processed_input_tn = input_processor(self.base_input_ph)

        if target_processor is None:
            target_processor = lambda x: default_target_processor(
                x, IO_num_hidden, z_dim, internal_nonlinearity)
            (processed_targets, target_processor_var,
             meta_class_processor_var) = target_processor(self.base_target_ph)

            if self.separate_targ_net:
                with tf.variable_scope("target_net", reuse=False):
                    (processed_targets_tn, target_processor_var_tn,
                     meta_class_processor_var_tn) = target_processor(self.base_target_ph)
        else:
            if output_processor is None or meta_class_processor is None:
                raise ValueError("You cannot use the default output or "
                                 "meta_class processor without the default "
                                 "target processor.")
            processed_targets = target_processor(self.base_target_ph)
            if self.separate_targ_net:
                with tf.variable_scope("target_net", reuse=False):
                    processed_targets_tn = target_processor(self.base_target_ph)

        if output_processor is None:
            output_processor = lambda x: default_output_processor(
                x, target_processor_var)
            if self.separate_targ_net:
                output_processor_tn = lambda x: default_output_processor(
                    x, target_processor_var_tn)

        if outcome_shape is not None and outcome_processor is None:
            outcome_processor = lambda x: default_outcome_processor(
                x, IO_num_hidden, z_dim, internal_nonlinearity)
            processed_outcomes = outcome_processor(self.base_outcome_ph)
            if self.separate_targ_net:
                with tf.variable_scope("target_net", reuse=False):
                    processed_outcomes_tn = outcome_processor(self.base_outcome_ph)

        ## Meta: (Input, Output) -> Z (function embedding)
        self.task_index_ph = tf.placeholder(tf.int32, [None,])  # if using persistent task reps

        self.meta_input_indices_ph = tf.placeholder(tf.int32, shape=[None, ])
        self.meta_target_indices_ph = tf.placeholder(tf.int32, [None,])

        self.meta_class_ph = tf.placeholder(tf.float32, shape=[None, 1]) # for meta classification tasks

        if meta_class_processor is None:
            meta_class_processor = lambda x: default_meta_class_processor(
                x, meta_class_processor_var)
        
        processed_class = meta_class_processor(self.meta_class_ph) 
        if self.separate_targ_net:
            with tf.variable_scope("target_net", reuse=False):
                processed_class = meta_class_processor(self.meta_class_ph) 


        # function embedding "guessing" network / meta network
        # {(emb_in, emb_out), ...} -> emb
        self.guess_input_mask_ph = tf.placeholder(tf.bool, shape=[None]) # which datapoints get excluded from the guess
        self.meta_batch_size = architecture_config["meta_batch_size"]  # this is used for generating guess masks

        def _meta_network(embedded_inputs, embedded_targets,
                          guess_mask=self.guess_input_mask_ph, reuse=True):
            num_hidden_meta = architecture_config["M_num_hidden"]
            with tf.variable_scope('meta', reuse=reuse):
                meta_input = tf.concat([embedded_inputs,
                                        embedded_targets], axis=-1)
                meta_input = tf.boolean_mask(meta_input,
                                             guess_mask)

                mh_1 = slim.fully_connected(meta_input, num_hidden_meta,
                                            activation_fn=internal_nonlinearity)
                mh_2 = slim.fully_connected(mh_1, num_hidden_meta,
                                            activation_fn=internal_nonlinearity)
                if architecture_config["M_max_pool"]:
                    mh_2b = tf.reduce_max(mh_2, axis=0, keep_dims=True)
                else:
                    mh_2b = tf.reduce_mean(mh_2, axis=0, keep_dims=True)

                mh_3 = slim.fully_connected(mh_2b, num_hidden_meta,
                                            activation_fn=internal_nonlinearity)

                guess_embedding = slim.fully_connected(mh_3, architecture_config["z_dim"],
                                                       activation_fn=None)
                return guess_embedding

        if outcome_shape is not None:  # outcomes seen by M != targets
            self.base_guess_emb = _meta_network(self.processed_input,
                                                processed_outcomes,
                                                reuse=False)
            if self.separate_targ_net:
                with tf.variable_scope("target_net", reuse=False):
                    self.base_guess_emb_tn = _meta_network(self.processed_input_tn,
                                                           processed_outcomes_tn,
                                                           reuse=False)

        else:
            self.base_guess_emb = _meta_network(self.processed_input,
                                                processed_targets,
                                                reuse=False)
            if self.separate_targ_net:
                with tf.variable_scope("target_net", reuse=False):
                    self.base_guess_emb_tn = _meta_network(self.processed_input_tn,
                                                           processed_targets_tn,
                                                           reuse=False)
        # TODO: Use it or lose it
        # for combination tasks 
#        def _combine_inputs(inputs_1, inputs_2, reuse=True):
#            with tf.variable_scope('meta/bi_combination', reuse=reuse):
#                c_input = tf.concat([inputs_1,
#                                     inputs_2], axis=-1)
#                c_input = tf.nn.dropout(c_input, self.keep_prob_ph)
#
#                ch_1 = slim.fully_connected(c_input, num_hidden_hyper,
#                                            activation_fn=internal_nonlinearity)
#                ch_1 = tf.nn.dropout(ch_1, self.keep_prob_ph)
#                ch_2 = slim.fully_connected(ch_1, num_hidden_hyper,
#                                            activation_fn=internal_nonlinearity)
#                ch_2 = tf.nn.dropout(ch_2, self.keep_prob_ph)
#                ch_3 = slim.fully_connected(ch_2, num_hidden_hyper,
#                                            activation_fn=internal_nonlinearity)
#                combined = tf.nn.dropout(ch_3, self.keep_prob_ph)
#                # also include direct inputs averaged
#                self.combination_weight = tf.get_variable('combination_weight',
#                                                          shape=[],
#                                                          initializer=tf.constant_initializer(0.))
#                c_w = tf.nn.sigmoid(self.combination_weight) 
#                combined = (c_w) * combined + (1.- c_w) * 0.5 * (inputs_1 + inputs_2) 
#                return combined 
#
#        self.combined_meta_inputs = _combine_inputs(self.meta_input_ph,
#                                                    self.meta_input_2_ph,
#                                                    reuse=False)
#
#        self.guess_meta_bf_function_emb = _meta_network(
#            self.combined_meta_inputs, self.meta_target_ph)


        ## language processing: lang -> Z
        if self.run_config["train_language"]:
            warnings.warn("Language has not been tested in this version of the code, some debugging may be needed")
            num_hidden_language = architecture_config["num_hidden_language"]
            num_lstm_layers = architecture_config["num_lstm_layers"]

            self.lang_keep_ph = lang_keep_ph = tf.placeholder(tf.float32)
            self.lang_keep_prob = 1. - architecture_config["lang_drop_prob"]
            self.language_input_ph = tf.placeholder(
                tf.int32, shape=[1, self.max_sentence_len])

            def _language_network(language_input, reuse=False):
                """Maps from language to a function embedding"""
                with tf.variable_scope("word_embeddings", reuse=reuse):
                    self.word_embeddings = tf.get_variable(
                        "embeddings", shape=[self.vocab_size, num_hidden_language])

                embedded_language = tf.nn.embedding_lookup(self.word_embeddings,
                                                           language_input)
                with tf.variable_scope("language_processing"):
                    cells = [tf.nn.rnn_cell.LSTMCell(
                        num_hidden_language) for _ in range(num_lstm_layers)]
                    stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

                    state = stacked_cell.zero_state(1, dtype=tf.float32)

                    for i in range(self.max_sentence_len):
                        this_input = embedded_language[:, i, :]
                        this_input = tf.nn.dropout(this_input,
                                                   lang_keep_ph)
                        cell_output, state = stacked_cell(this_input, state)

                    cell_output = tf.nn.dropout(cell_output,
                                               lang_keep_ph)
                    language_hidden = slim.fully_connected(
                        cell_output, num_hidden_language,
                        activation_fn=internal_nonlinearity)

                    language_hidden = tf.nn.dropout(language_hidden,
                                               lang_keep_ph)

                    func_embeddings = slim.fully_connected(language_hidden,
                                                           num_hidden_hyper,
                                                           activation_fn=None)
                return func_embeddings

            self.language_function_emb = _language_network(self.language_input_ph,
                                                           reuse=False)
            if self.separate_targ_net:
                with tf.variable_scope("target_net", reuse=False):
                    self.language_function_emb_tn = _language_network(self.language_input_ph,
                                                                      reuse=False)


        ## persistent/cached embeddings
        # if not architecture_config["persistent_task_reps"], these will just be updated
        # manually. Either way, they are used for eval + meta tasks.
        with tf.variable_scope("persistent_cached"):
            self.persistent_embeddings = tf.get_variable(
                "cached_task_embeddings",
                [self.num_tasks,
                 architecture_config["z_dim"]],
                dtype=tf.float32)

            self.update_persistent_embeddings_ph = tf.placeholder(
                tf.float32,
                [None, architecture_config["z_dim"]])

            self.update_embeddings = tf.scatter_update(
                self.persistent_embeddings,
                self.task_index_ph,
                self.update_persistent_embeddings_ph)

        if self.separate_targ_net:
            with tf.variable_scope("target_net", reuse=False):
                with tf.variable_scope("persistent_cached"):
                    self.persistent_embeddings_tn = tf.get_variable(
                        "cached_task_embeddings",
                        [self.num_tasks,
                         architecture_config["z_dim"]],
                        dtype=tf.float32)

        # don't update cached embeddings if not persistent -- will be updated manually
        if not architecture_config["persistent_task_reps"]:
            self.persistent_embeddings = tf.stop_gradient(self.persistent_embeddings)

        if self.separate_targ_net:
            def _get_persistent_embeddings(task_indices, target_net=False):
                if target_net:
                    persistent_embs = self.persistent_embeddings_tn

                else:
                    persistent_embs = self.persistent_embeddings

                return tf.nn.embedding_lookup(persistent_embs,
                                              task_indices)

        else:
            def _get_persistent_embeddings(task_indices):
                persistent_embs = self.persistent_embeddings

                return tf.nn.embedding_lookup(persistent_embs,
                                              task_indices)

        self.lookup_cached_emb = _get_persistent_embeddings(
            self.task_index_ph)

        if self.separate_targ_net:
            self.lookup_cached_emb_tn = _get_persistent_embeddings(
                self.task_index_ph, target_net=True)

        if architecture_config["persistent_task_reps"]:
            def _get_combined_embedding_and_match_loss(guess_embedding, task_index,
                                                       guess_weight,
                                                       target_net=False):
                cached_embedding = _get_persistent_embeddings(task_index)
                if guess_weight == "varied":
                    guess_weight = tf.random_uniform([], dtype=tf.float32)

                combined_embedding = guess_weight * guess_embedding + (1. - guess_weight) * cached_embedding
                emb_match_loss = architecture_config["emb_match_loss_weight"] * tf.nn.l2_loss(
                    guess_embedding - cached_embedding)
                return combined_embedding, emb_match_loss

            (self.base_combined_emb,
             self.base_emb_match_loss) = _get_combined_embedding_and_match_loss(
                self.base_guess_emb, self.task_index_ph,
                architecture_config["combined_emb_guess_weight"])
            if self.separate_targ_net:
                (self.base_combined_emb_tn,
                 _) = _get_combined_embedding_and_match_loss(
                    self.base_guess_emb_tn, self.task_index_ph,
                    architecture_config["combined_emb_guess_weight"])

        meta_input_embeddings = _get_persistent_embeddings(
            self.meta_input_indices_ph)
        meta_target_embeddings = _get_persistent_embeddings(
            self.meta_target_indices_ph)

        self.meta_input_embeddings = meta_input_embeddings 
        self.meta_target_embeddings = meta_target_embeddings 

        self.meta_class_guess_emb = _meta_network(meta_input_embeddings,
                                                  processed_class)
        self.meta_map_guess_emb = _meta_network(meta_input_embeddings,
                                                meta_target_embeddings)


        if architecture_config["persistent_task_reps"]:
            (self.meta_class_combined_emb,
             self.meta_class_emb_match_loss) = _get_combined_embedding_and_match_loss(
                self.meta_class_guess_emb, self.task_index_ph,
                architecture_config["combined_emb_guess_weight"])

            (self.meta_map_combined_emb,
             self.meta_map_emb_match_loss) = _get_combined_embedding_and_match_loss(
                self.meta_map_guess_emb, self.task_index_ph,
                architecture_config["combined_emb_guess_weight"])
        else:
            self.base_combined_emb = self.base_guess_emb
            if self.separate_targ_net:
                self.base_combined_emb_tn = self.base_guess_emb_tn
            self.meta_class_combined_emb = self.meta_class_guess_emb
            self.meta_map_combined_emb = self.meta_map_guess_emb


        ## hyper_network: Z -> (F: Z -> Z)
        self.feed_embedding_ph = tf.placeholder(np.float32,
                                                [1, z_dim])

        z_dim = architecture_config["z_dim"]
        num_hidden_hyper = architecture_config["H_num_hidden"]
        num_hidden_F = architecture_config["F_num_hidden"]
        num_task_hidden_layers = architecture_config["F_num_hidden_layers"]

        tw_range = architecture_config["task_weight_weight_mult"]/np.sqrt(
            num_hidden_F * num_hidden_hyper) # yields a very very roughly O(1) map
        task_weight_gen_init = tf.random_uniform_initializer(-tw_range,
                                                             tw_range)

        def _hyper_network(function_embedding, reuse=True):
            with tf.variable_scope('hyper', reuse=reuse):
                hyper_hidden = function_embedding
                for _ in range(architecture_config["H_num_hidden_layers"]):
                    hyper_hidden = slim.fully_connected(hyper_hidden, num_hidden_hyper,
                                                        activation_fn=internal_nonlinearity)

                hidden_weights = []
                hidden_biases = []

                task_weights = slim.fully_connected(hyper_hidden, num_hidden_F*(z_dim +(num_task_hidden_layers-1)*num_hidden_F + z_dim),
                                                    activation_fn=None,
                                                    weights_initializer=task_weight_gen_init)

                task_weights = tf.reshape(task_weights, [-1, num_hidden_F, (z_dim + (num_task_hidden_layers-1)*num_hidden_F + z_dim)])
                task_biases = slim.fully_connected(hyper_hidden, num_task_hidden_layers * num_hidden_F + z_dim,
                                                   activation_fn=None)

                Wi = tf.transpose(task_weights[:, :, :z_dim], perm=[0, 2, 1])
                bi = task_biases[:, :num_hidden_F]
                hidden_weights.append(Wi)
                hidden_biases.append(bi)
                for i in range(1, num_task_hidden_layers):
                    Wi = tf.transpose(task_weights[:, :, z_dim+(i-1)*num_hidden_F:z_dim+i*num_hidden_F], perm=[0, 2, 1])
                    bi = task_biases[:, num_hidden_F*i:num_hidden_F*(i+1)]
                    hidden_weights.append(Wi)
                    hidden_biases.append(bi)
                Wfinal = task_weights[:, :, -z_dim:]
                bfinal = task_biases[:, -z_dim:]

                for i in range(num_task_hidden_layers):
                    hidden_weights[i] = tf.squeeze(hidden_weights[i], axis=0)
                    hidden_biases[i] = tf.squeeze(hidden_biases[i], axis=0)

                Wfinal = tf.squeeze(Wfinal, axis=0)
                bfinal = tf.squeeze(bfinal, axis=0)
                hidden_weights.append(Wfinal)
                hidden_biases.append(bfinal)
                return hidden_weights, hidden_biases

        self.base_task_params = _hyper_network(self.base_combined_emb,
                                               reuse=False)
        if self.separate_targ_net:
            with tf.variable_scope("target_net", reuse=False):
                self.base_task_params_tn = _hyper_network(self.base_combined_emb_tn,
                                                       reuse=False)

        self.meta_map_task_params = _hyper_network(self.meta_map_combined_emb)
        self.meta_class_task_params = _hyper_network(self.meta_class_combined_emb)

        if self.run_config["train_language"]:
            self.lang_task_params = _hyper_network(self.language_function_emb)

        self.fed_emb_task_params = _hyper_network(self.feed_embedding_ph)
        self.cached_emb_task_params = _hyper_network(self.lookup_cached_emb)
        if self.separate_targ_net:
            with tf.variable_scope("target_net", reuse=True):
                self.cached_emb_task_params_tn = _hyper_network(self.lookup_cached_emb_tn, reuse=True)

        ## task network F: Z -> Z
        def _task_network(task_params, processed_input):
            hweights, hbiases = task_params
            task_hidden = processed_input
            for i in range(num_task_hidden_layers):
                task_hidden = internal_nonlinearity(
                    tf.matmul(task_hidden, hweights[i]) + hbiases[i])

            raw_output = tf.matmul(task_hidden, hweights[-1]) + hbiases[-1]

            return raw_output

        self.base_raw_output = _task_network(self.base_task_params,
                                             self.processed_input)
        self.base_output = output_processor(self.base_raw_output)
        if self.separate_targ_net:
            self.base_raw_output_tn = _task_network(self.base_task_params_tn,
                                                    self.processed_input_tn)
            self.base_output_tn = output_processor_tn(self.base_raw_output_tn)

        self.base_raw_fed_emb_output = _task_network(self.fed_emb_task_params,
                                                     self.processed_input)
        self.base_fed_emb_output = output_processor(
            self.base_raw_fed_emb_output)

        self.meta_class_raw_output = _task_network(self.meta_class_task_params,
                                                   meta_input_embeddings)
        self.meta_class_output_logits = tf.matmul(
            self.meta_class_raw_output, meta_class_processor_var)
        self.meta_class_output = tf.nn.sigmoid(self.meta_class_output_logits)

        self.meta_map_output = _task_network(self.meta_map_task_params,
                                             meta_input_embeddings) 

        self.meta_map_fed_emb_output = _task_network(self.fed_emb_task_params,
                                                     meta_input_embeddings) 

        self.base_cached_emb_raw_output = _task_network(
            self.cached_emb_task_params, self.processed_input)
        self.base_cached_emb_output = output_processor(
            self.base_cached_emb_raw_output)
        if self.separate_targ_net:
            self.base_cached_emb_raw_output_tn = _task_network(
                self.cached_emb_task_params_tn, self.processed_input_tn)
            self.base_cached_emb_output_tn = output_processor(
                self.base_cached_emb_raw_output_tn)

        self.meta_cached_emb_raw_output = _task_network(
            self.cached_emb_task_params, meta_input_embeddings)
        self.meta_class_cached_emb_output_logits = tf.matmul(
            self.meta_cached_emb_raw_output, meta_class_processor_var)

        if self.run_config["train_language"]:
            self.base_lang_raw_output = _task_network(self.lang_task_params,
                                                     processed_input)
            self.base_lang_output = output_processor(self.base_lang_raw_output)

            self.meta_map_lang_output = _task_network(self.lang_task_params,
                                                      meta_input_embeddings) 

        if self.architecture_config["output_masking"]:
            # E.g. in RL we have to mask base output in loss because can only
            # learn about the action actually taken 
            self.base_target_mask_ph = tf.placeholder(
                tf.bool, shape=[None] + output_shape)

            self.base_unmasked_output = self.base_output
            self.base_output = tf.boolean_mask(self.base_unmasked_output,
                                               self.base_target_mask_ph)

            self.base_fed_emb_unmasked_output = self.base_fed_emb_output
            self.base_fed_emb_output = tf.boolean_mask(self.base_fed_emb_unmasked_output,
                                               self.base_target_mask_ph)

            self.base_cached_emb_unmasked_output = self.base_cached_emb_output
            self.base_cached_emb_output = tf.boolean_mask(self.base_cached_emb_unmasked_output,
                                                          self.base_target_mask_ph)

            self.base_unmasked_targets = self.base_target_ph
            self.base_targets = tf.boolean_mask(self.base_unmasked_targets,
                                                self.base_target_mask_ph)
        else: 
            self.base_targets = self.base_target_ph


        # allow any task-specific alterations before computing the losses
        self._pre_loss_calls() 

        ## Losses 

        if base_loss is None:
            base_loss = default_base_loss
        if meta_loss is None:
            meta_loss = default_meta_loss
        if meta_class_loss is None:
            meta_class_loss = default_meta_class_loss

        self.total_base_loss = base_loss(self.base_output, self.base_targets) 


        self.total_base_fed_emb_loss = base_loss(self.base_fed_emb_output,
                                                 self.base_targets) 

        self.total_meta_class_loss = meta_class_loss(
            self.meta_class_output_logits, self.meta_class_ph) 

        self.total_meta_map_loss = meta_loss(self.meta_map_output,
                                             meta_target_embeddings) 

        self.total_meta_map_fed_emb_loss = meta_loss(self.meta_map_fed_emb_output,
                                                     meta_target_embeddings) 

        self.total_base_cached_emb_loss = base_loss(
            self.base_cached_emb_output, self.base_targets) 

        self.total_meta_class_cached_emb_loss = meta_class_loss(
            self.meta_class_cached_emb_output_logits, self.meta_class_ph) 

        self.total_meta_map_cached_emb_loss = meta_loss(
            self.meta_cached_emb_raw_output, meta_target_embeddings) 

        if self.architecture_config["persistent_task_reps"]: # Add the emb_match losses
            self.total_base_loss += self.base_emb_match_loss
            self.total_meta_class_loss += self.meta_class_emb_match_loss
            self.total_meta_map_loss += self.meta_map_emb_match_loss

        if self.run_config["train_language"]:
            self.total_base_lang_loss = base_loss(self.base_lang_output,
                                                  self.base_targets) 

            self.total_meta_map_lang_loss = meta_loss(self.meta_map_lang_output,
                                                      meta_target_embeddings) 

        if architecture_config["optimizer"] == "Adam":
            optimizer = tf.train.AdamOptimizer(self.lr_ph)
        elif architecture_config["optimizer"] == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(self.lr_ph)
        else:
            raise ValueError("Unknown optimizer: %s" % architecture_config["optimizer"])

        self.base_train_op = optimizer.minimize(self.total_base_loss)
        self.meta_class_train_op = optimizer.minimize(self.total_meta_class_loss)
        self.meta_map_train_op = optimizer.minimize(self.total_meta_map_loss)

        if self.run_config["train_language"]:
            self.base_lang_train_op = optimizer.minimize(self.total_base_lang_loss)
            self.meta_map_lang_train_op = optimizer.minimize(self.total_meta_map_lang_loss)

        if self.separate_targ_net:
            learner_vars = [v for v in tf.trainable_variables() if "target_net" not in v.name]
            target_vars = [v for v in tf.trainable_variables() if "target_net" in v.name]

            # there may be extra vars in the learner, e.g. because language is in learner but not target
            matched_learner_vars = [v for v_targ in target_vars for v in learner_vars if v.name == "/".join(v_targ.name.split("/")[1:])]  

            ## copy learner to target
            self.update_target_network_op = [v_targ.assign(v) for v_targ, v in zip(target_vars, matched_learner_vars)]


    def _pre_loss_calls(self):
        """Called after generating the outputs, but before generating losses."""
        pass

    def _sess_and_init(self):
        # Saver
        self.saver = tf.train.Saver()

        # initialize
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())
        self.fill_buffers(num_data_points=self.architecture_config["memory_buffer_size"])
        if self.separate_targ_net:
            self.sess.run(self.update_target_network_op)

        save_config(self.run_config["run_config_filename"], self.run_config) 
        save_config(self.run_config["architecture_config_filename"],
                    self.architecture_config) 


    def get_new_memory_buffer(self):
        """Can be overriden by child"""
        return memory_buffer(length=self.architecture_config["memory_buffer_size"],
                             input_width=self.architecture_config["input_shape"][0],
                             outcome_width=self.architecture_config["output_shape"][0])

    def base_task_lookup(self, task):
        if isinstance(task, str):
            task_name = task
        else:
            task_name = str(task)

        if task_name not in self.task_indices:
            self.memory_buffers[task_name] = self.get_new_memory_buffer() 
            self.task_indices[task_name] = self.num_tasks
            self.num_tasks += 1

        return (task_name,
                self.memory_buffers[task_name],
                self.task_indices[task_name])

    def meta_task_lookup(self, task):
        if task not in self.task_indices:
            self.meta_datasets[task] = {} 
            self.task_indices[task] = self.num_tasks
            self.num_tasks += 1

        return (task,
                self.meta_datasets[task],
                self.task_indices[task])

    def fill_buffers(self, num_data_points=1):
        """Add new "experiences" to memory buffers."""
        raise NotImplementedError("fill_buffers() should be overridden by the child class!")

    def _random_guess_mask(self, dataset_length, meta_batch_size=None):
        if meta_batch_size is None:
            meta_batch_size = self.meta_batch_size
        mask = np.zeros(dataset_length, dtype=np.bool)
        indices = np.random.permutation(dataset_length)[:meta_batch_size]
        mask[indices] = True
        return mask

    def sample_from_memory_buffer(self, memory_buffer):
        """Return experiences from the memory buffer.
        
        The results are expected to be a set of inputs and targets, so this
        will likely need to be overridden for other settings such as RL. 
        """
        input_buff, output_buff = memory_buffer.get_memories()
        return input_buff, output_buff


    def intify_task(self, task): 
        """Will need to be overridden"""
        raise NotImplementedError("intify task should be overridden by the child class!")

    def build_feed_dict(self, task, lr=None, fed_embedding=None,
                        call_type="base_standard_train"):
        """Build a feed dict.

        This function should be overridden by the child class if the necessary.
        
        Args:
            task: the task def.
            lr: the learning rate if training.
            fed_embedding: embedding to use for performing the task, if any.
            call_type: one of "base_standard_train", "base_lang_train",
                "base_lang_eval", "base_fed_eval", "base_cached_eval",
                "metaclass_standard_train", "metamap_standard_train", 
                "metaclass_cached_eval", "metamap_cached_eval", 
                "metaclass_lang_train", "metamap_lang_train", 
                "metaclass_lang_eval", "metamap_lang_eval"
        """
        feed_dict = {}

        base_or_meta, call_type, train_or_eval = call_type.split("_")

        if base_or_meta == "base":
            task_name, memory_buffer, task_index = self.base_task_lookup(task)
            inputs, outputs = self.sample_from_memory_buffer(memory_buffer)
            feed_dict[self.base_input_ph] = inputs
            feed_dict[self.base_target_ph] = outputs
            feed_dict[self.guess_input_mask_ph] = self._random_guess_mask(
                len(outputs))
        else: # meta call
            task_name, meta_dataset, task_index = self.meta_task_lookup(task)
            meta_input_indices = meta_dataset[train_or_eval]["in"]
            feed_dict[self.meta_input_indices_ph] = meta_input_indices
            if train_or_eval == "train":
                if len(meta_input_indices) < self.meta_batch_size:
                    meta_batch_size = len(meta_input_indices) // 2
                else:
                    meta_batch_size = self.meta_batch_size
                guess_mask = self._random_guess_mask(
                    len(meta_input_indices), meta_batch_size=meta_batch_size)
            else:
                guess_mask = meta_dataset["eval"]["gm"]  # eval on the right tasks
            feed_dict[self.guess_input_mask_ph] = guess_mask 
            meta_targets = meta_dataset[train_or_eval]["out"]
            if base_or_meta == "metamap":
                feed_dict[self.meta_target_indices_ph] = meta_targets
            else:
                feed_dict[self.meta_class_ph] = meta_targets

        if call_type == "fed":
            if len(fed_embedding.shape) == 1:
                fed_embedding = np.expand_dims(fed_embedding, axis=0)
            feed_dict[self.feed_embedding_ph] = fed_embedding
        elif call_type == "lang":
            feed_dict[self.language_input_ph] = self.intify_task(task_name)
        if call_type == "cached" or self.architecture_config["persistent_task_reps"]:
            feed_dict[self.task_index_ph] = [task_index]

        if train_or_eval == "train":
            feed_dict[self.lr_ph] = lr
            feed_dict[self.keep_prob_ph] = self.tkp
            if call_type == "lang":
                feed_dict[self.lang_keep_prob_ph] = self.lang_keep_prob
        else:
            feed_dict[self.keep_prob_ph] = 1.
            if call_type == "lang":
                feed_dict[self.lang_keep_prob_ph] = 1. 

        return feed_dict

    def base_train_step(self, task, lr):
        feed_dict = self.build_feed_dict(task, lr=lr, call_type="base_standard_train")
        self.sess.run(self.base_train_op, feed_dict=feed_dict)

    def base_language_train_step(self, task, lr):
        feed_dict = self.build_feed_dict(task, lr=lr, call_type="base_lang_train")
        self.sess.run(self.base_lang_train_op, feed_dict=feed_dict)

    def base_eval(self, task, train_or_eval):
        feed_dict = self.build_feed_dict(task, call_type="base_cached_eval")
        fetches = [self.total_base_loss]
        res = self.sess.run(fetches, feed_dict=feed_dict)
        name = str(task) + ":" + train_or_eval
        return [name], res

    def run_base_eval(self):
        """Run evaluation on basic tasks."""
        names = []
        losses = [] 
        for task_set, train_or_eval in zip(
            [self.base_train_tasks, self.base_eval_tasks],
            ["train", "eval"]):
            for task in task_set:
                these_names, these_losses = self.base_eval(task, train_or_eval)
                names += these_names
                losses += these_losses

        return names, losses
        
    def base_language_eval(self, task):
        feed_dict = self.build_feed_dict(task, call_type="base_lang_eval")
        fetches = [self.total_base_lang_loss]
        res = self.sess.run(fetches, feed_dict=feed_dict)
        name = str(task) + ":" + train_or_eval
        return res

    def run_base_language_eval(self):
        losses = [] 
        names = []
        for task_set, train_or_eval in zip(
            [self.base_train_tasks, self.base_eval_tasks],
            ["train", "eval"]):
            for task in task_set:
                these_names, these_losses = self.base_language_eval(task, train_or_eval)
                names += these_names
                losses += these_losses

        return names, losses

    def base_embedding_eval(self, embedding, task):
        feed_dict = self.build_feed_dict(task, fed_embedding=embedding, call_type="base_fed_eval")
        fetches = [self.total_base_fed_emb_loss]
        res = self.sess.run(fetches, feed_dict=feed_dict)
        return res

    def get_base_guess_embedding(self, task):
        feed_dict = self.build_feed_dict(task, call_type="base_standard_eval")
        res = self.sess.run(self.base_guess_emb, feed_dict=feed_dict)
        return res

    def get_language_embedding(self, intified_task):
        feed_dict = self.build_feed_dict(task, call_type="base_lang_eval")
        res = self.sess.run(self.language_function_emb, feed_dict=feed_dict)
        return res

    def get_base_cached_embedding(self, task):
        _, _, task_index = self.base_task_lookup(task)
        feed_dict = {self.task_index_ph: [task_index]} 
        res = self.sess.run(self.lookup_cached_emb, feed_dict=feed_dict)
        return res

    def meta_class_loss_eval(self, meta_task):
        feed_dict = self.build_feed_dict(meta_task, call_type="metaclass_cached_eval")
        return self.sess.run(self.total_meta_class_cached_emb_loss, feed_dict=feed_dict)
        
    def meta_map_loss_eval(self, meta_task):
        feed_dict = self.build_feed_dict(meta_task, call_type="metamap_cached_eval")
        return self.sess.run(self.total_meta_map_cached_emb_loss, feed_dict=feed_dict)

    def run_meta_loss_eval(self):
        names = []
        losses = []
        for meta_tasks, meta_class, train_or_eval in zip(
                [self.meta_class_train_tasks,
                 self.meta_class_eval_tasks,
                 self.meta_map_train_tasks,
                 self.meta_map_eval_tasks],
                [True, True, False, False],
                ["train", "eval", "train", "eval"]):
            for meta_task in meta_tasks:
                if meta_class:
                    loss = self.meta_class_loss_eval(meta_task)
                else:
                    loss = self.meta_map_loss_eval(meta_task)
                names.append(meta_task + ":" + train_or_eval)
                losses.append(loss)

        return names, losses

    def get_meta_guess_embedding(self, meta_task, meta_class):
        """Note: cached base embeddings must be up to date!"""
        call_type = "metaclass_standard_eval" if meta_class else "metamap_standard_eval"
        feed_dict = self.build_feed_dict(meta_task, call_type=call_type)
        if meta_class:
            fetch = self.meta_class_guess_emb 
        else:
            fetch = self.meta_map_guess_emb 

        return self.sess.run(fetch, feed_dict=feed_dict)

    def meta_true_eval_step(self, meta_mapping, meta_mapping_train_or_eval):
        meta_dataset = self.meta_datasets[meta_mapping]["eval"]

        feed_dict = self.build_feed_dict(meta_mapping,
                                         call_type="metamap_cached_eval")
        result_embeddings = self.sess.run(self.meta_map_output,
                                          feed_dict=feed_dict) 

        names = []
        losses = []
        i = 0

        for train_or_eval in ["train", "eval"]:
            for (task, other) in self.meta_pairings[meta_mapping][train_or_eval]:
                mapped_embedding = result_embeddings[i, :]

                names.append(
                    meta_mapping + ":metamapping_is_" + meta_mapping_train_or_eval + ":example_is_" + train_or_eval + ":" + task + "->" + other)
                this_loss = self.base_embedding_eval(embedding=mapped_embedding,
                                                     task=other)[0]
                losses.append(this_loss)
                i = i + 1
        return names, losses
    
    def run_meta_true_eval(self):
        """Evaluates true meta loss, i.e. the accuracy of the model produced
           by the embedding output by the meta task"""
        names = []
        losses = []
        for these_meta_mappings, train_or_eval in zip([self.meta_map_train_tasks,
                                                       self.meta_map_eval_tasks],
                                                      ["train", "eval"]):
            for meta_mapping in these_meta_mappings:
                these_names, these_losses = self.meta_true_eval_step(
                    meta_mapping, train_or_eval) 
                names += these_names
                losses += these_losses

        return names, losses 

    def meta_class_train_step(self, meta_task, meta_lr):
        feed_dict = self.build_feed_dict(meta_task, lr=meta_lr, call_type="metaclass_standard_train")
        self.sess.run(self.meta_class_train_op, feed_dict=feed_dict)

    def meta_map_train_step(self, meta_task, meta_lr):
        feed_dict = self.build_feed_dict(meta_task, lr=meta_lr, call_type="metamap_standard_train")
        self.sess.run(self.meta_map_train_op, feed_dict=feed_dict)

    def update_base_task_embeddings(self):
        """Updates cached embeddings (for use if embeddings are not persistent)"""
        if self.architecture_config["persistent_task_reps"]:
            warnings.warn("Overwriting persistent embeddings... Is this "
                          "intentional?")

        update_inds = []
        update_values = []
        for task in self.base_train_tasks + self.base_eval_tasks:
            task_emb = self.get_base_guess_embedding(task)
            _, _, task_index = self.base_task_lookup(task)
            update_inds.append(task_index)
            update_values.append(task_emb[0])

        self.sess.run(
            self.update_embeddings,
            feed_dict={
                self.task_index_ph: np.array(update_inds, dtype=np.int32),
                self.update_persistent_embeddings_ph: update_values
            })

    def update_meta_task_embeddings(self):
        if self.architecture_config["persistent_task_reps"]:
            warnings.warn("Overwriting persistent embeddings... Is this "
                          "intentional?")

        update_inds = []
        update_values = []
        for these_tasks, meta_class in zip([self.meta_class_train_tasks + self.meta_class_eval_tasks,
                                            self.meta_map_train_tasks + self.meta_map_eval_tasks],
                                           [True, False]):
            for task in these_tasks:
                task_emb = self.get_meta_guess_embedding(task, meta_class)
                _, _, task_index = self.meta_task_lookup(task)
                update_inds.append(task_index)
                update_values.append(task_emb[0])

        self.sess.run(
            self.update_embeddings,
            feed_dict={
                self.task_index_ph: np.array(update_inds, dtype=np.int32),
                self.update_persistent_embeddings_ph: update_values
            })

    def save_parameters(self, filename):
        self.saver.save(self.sess, filename)

    def restore_parameters(self, filename):
        self.saver.restore(self.sess, filename)

    def run_eval(self, epoch, print_losses=True):
        train_meta = self.run_config["train_meta"]

        base_names, base_losses = self.run_base_eval()
        
        if train_meta:
            meta_names, meta_losses = self.run_meta_loss_eval()
            meta_true_names, meta_true_losses = self.run_meta_true_eval()
        else:
            meta_names, meta_losses = [], []

        if epoch == 0:
            # set up format string
            self.loss_format = ", ".join(["%f" for _ in base_names + meta_names]) + "\n"

            # write headers and overwrite existing file 
            with open(self.run_config["loss_filename"], "w") as fout:
                fout.write("epoch, " + ", ".join(base_names + meta_names) + "\n")

            if train_meta:
                self.meta_true_loss_format = ", ".join(["%f" for _ in meta_true_names]) + "\n"
                with open(self.run_config["meta_filename"], "w") as fout:
                    fout.write("epoch, " + ", ".join(meta_true_names) + "\n")

        epoch_s = "%i, " % epoch
        with open(self.run_config["loss_filename"], "a") as fout:
            formatted_losses = epoch_s + (self.loss_format % tuple(
                base_losses + meta_losses))
            fout.write(formatted_losses)

        if print_losses:
            print(formatted_losses)

        if train_meta:
            with open(self.run_config["meta_filename"], "a") as fout:
                formatted_losses = epoch_s + (self.meta_true_loss_format % tuple(
                    meta_true_losses))
                fout.write(formatted_losses)

        if self.run_config["train_language"]:
            lang_names, lang_losses = self.run_lang_eval()
            if epoch == 0:
                self.lang_loss_format = ", ".join(["%f" for _ in lang_names]) + "\n"
                with open(self.run_config["lang_filename"], "w") as fout:
                    fout.write("epoch, " + ", ".join(lang_names) + "\n")

            with open(self.run_config["lang_filename"], "a") as fout:
                formatted_losses = epoch_s + (self.lang_loss_format % tuple(
                    lang_losses))
                fout.write(formatted_losses)

    def end_epoch_calls(self, epoch): 
        """Can be overridden to change things like exploration over learning."""
        pass

    def run_training(self):
        """Train model."""
        train_language = self.run_config["train_language"]
        train_meta = self.run_config["train_meta"]

        eval_every = self.run_config["eval_every"]
        
        learning_rate = self.run_config["init_learning_rate"]
        language_learning_rate = self.run_config["init_language_learning_rate"]
        meta_learning_rate = self.run_config["init_meta_learning_rate"]
        
        num_epochs = self.run_config["num_epochs"]
        eval_every = self.run_config["eval_every"]
        refresh_mem_buffs_every = self.run_config["refresh_mem_buffs_every"]
        lr_decays_every = self.run_config["lr_decays_every"]
        lr_decay = self.run_config["lr_decay"]
        meta_lr_decay = self.run_config["meta_lr_decay"]
        min_learning_rate = self.run_config["min_learning_rate"]
        min_meta_learning_rate = self.run_config["min_meta_learning_rate"]

        if train_language:
            language_lr_decay = self.run_config["language_lr_decay"]
            min_language_learning_rate = self.run_config["min_language_learning_rate"]

        if not(self.architecture_config["persistent_task_reps"]):
            self.update_base_task_embeddings()  # make sure we're up to date
            if train_meta:
                self.update_meta_task_embeddings()

        self.run_eval(epoch=0)

        if train_meta:
            tasks = self.base_train_tasks + self.meta_class_train_tasks + self.meta_map_train_tasks
            task_types = ["base"] * len(self.base_train_tasks) + ["meta_class"] * len(self.meta_class_train_tasks) + ["meta_map"] * len(self.meta_map_train_tasks)
        else:
            tasks = self.base_train_tasks
            task_types = ["base"] * len(self.base_train_tasks)

        for epoch in range(1, num_epochs+1):
            if epoch % refresh_mem_buffs_every == 0:
                self.fill_buffers(num_data_points=self.architecture_config["memory_buffer_size"])

            order = np.random.permutation(len(tasks))
            for task_i in order:
                task = tasks[task_i]
                task_type = task_types[task_i]

                if task_type == "base":
                    self.base_train_step(task, learning_rate)
                    if train_language:
                        self.base_lang_train_step(task,
                                                  language_learning_rate)

                elif task_type == "meta_class":
                    self.meta_class_train_step(task, meta_learning_rate)
                    if train_language:
                        self.meta_class_lang_train_step(task, language_learning_rate)
                else: 
                    self.meta_map_train_step(task, meta_learning_rate)
                    if train_language:
                        self.meta_map_lang_train_step(task, language_learning_rate)

            if not(self.architecture_config["persistent_task_reps"]):
                self.update_base_task_embeddings()  # make sure we're up to date
                if train_meta:
                    self.update_meta_task_embeddings()

            if epoch % eval_every == 0:
                self.run_eval(epoch)

            if epoch % lr_decays_every == 0 and epoch > 0:
                if learning_rate > min_learning_rate:
                    learning_rate *= lr_decay

                if meta_learning_rate > min_meta_learning_rate:
                    meta_learning_rate *= meta_lr_decay

                if train_language and language_learning_rate > min_language_learning_rate:
                    language_learning_rate *= language_lr_decay

            self.end_epoch_calls(epoch)
