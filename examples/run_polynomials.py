from itertools import permutations
from copy import deepcopy

import numpy as np
import tensorflow as tf

from HoMM import HoMM_model
from HoMM.configs import default_run_config
import polynomials

run_config = default_run_config.default_run_config
run_config.update({
    "output_dir": "polynomials_results/",
    
    "num_base_train_tasks": 60, # prior to meta-augmentation
    "num_base_eval_tasks": 40, # prior to meta-augmentation

    "num_variables": 4,
    "max_degree": 2,
    "poly_coeff_sd": 2.5,
    "point_val_range": 1,

    "meta_add_vals": [-3, -1, 1, 3],
    "meta_mult_vals": [-3, -1, 3],
    "new_meta_tasks": [],
    "new_meta_mappings": ["add_%f" % 2., "add_%f" % -2., "mult_%f" % 2., "mult_%f" % -2.],
})

class poly_HoMM_model(HoMM_model.HoMM_model):
    def __init__(self, run_config=None):
        super(poly_HoMM_model, self).__init__(run_config=run_config)

    def _pre_build_calls(self):
        # set up the base tasks
        poly_fam = polynomials.polynomial_family(self.run_config["num_variables"], self.run_config["max_degree"])
        self.run_config["variables"] = poly_fam.variables

        self.base_train_tasks = [poly_fam.sample_polynomial(coefficient_sd=self.run_config["poly_coeff_sd"]) for _ in range(self.run_config["num_base_train_tasks"])]
        self.base_eval_tasks = [poly_fam.sample_polynomial(coefficient_sd=self.run_config["poly_coeff_sd"]) for _ in range(self.run_config["num_base_eval_tasks"])]

        # set up the meta tasks

        self.meta_class_train_tasks = ["is_constant_polynomial"] + ["is_intercept_nonzero"] + ["is_%s_relevant" % var for var in self.run_config["variables"]]
        self.meta_class_eval_tasks = [] 

        self.meta_map_train_tasks = ["square"] + ["add_%f" % c for c in self.run_config["meta_add_vals"]] + ["mult_%f" % c for c in self.run_config["meta_mult_vals"]]
        self.meta_map_eval_tasks = self.run_config["new_meta_mappings"] 

        permutation_mappings = ["permute_" + "".join([str(x) for x in p]) for p in permutations(range(self.run_config["num_variables"]))]
        np.random.seed(0)
        np.random.shuffle(permutation_mappings)
        self.meta_map_train_tasks += permutation_mappings[:len(permutation_mappings)//2]
        self.meta_map_eval_tasks += permutation_mappings[len(permutation_mappings)//2:]

        # set up language

        #vocab = ['PAD'] + [str(x) for x in range(10)] + [".", "+", "-", "^"] + poly_fam.variables
        #vocab_to_int = dict(zip(vocab, range(len(vocab))))
        #
        #self.run_config["vocab"] = vocab

        # set up the meta pairings 
        self.meta_pairings, implied_tasks_train_tasks, implied_tasks_eval_tasks = polynomials.get_meta_pairings(
            base_train_tasks=self.base_train_tasks,
            base_eval_tasks=self.base_eval_tasks,
            meta_class_train_tasks=self.meta_class_train_tasks,
            meta_class_eval_tasks=self.meta_class_eval_tasks,
            meta_map_train_tasks=self.meta_map_train_tasks,
            meta_map_eval_tasks=self.meta_map_eval_tasks) 

        # add the base tasks implied by the mappings
        self.base_train_tasks += implied_tasks_train_tasks
        self.base_eval_tasks += implied_tasks_eval_tasks

    def fill_buffers(self, num_data_points=1):
        """Add new "experiences" to memory buffers."""
        this_tasks = self.base_train_tasks + self.base_eval_tasks 
        for t in this_tasks:
            buff = self.memory_buffers[polynomials.stringify_polynomial(t)]
            x_data = np.zeros([num_data_points] + self.architecture_config["input_shape"])
            y_data = np.zeros([num_data_points] + self.architecture_config["output_shape"])
            for point_i in range(num_data_points):
                point = t.family.sample_point(val_range=self.run_config["point_val_range"])
                x_data[point_i, :] = point
                y_data[point_i, :] = t.evaluate(point)
            buff.insert(x_data, y_data)

    def intify_task(self, task):
        if task == "square":
            return [vocab_to_int[task]]
        elif task[:3] == "add":
            val = str(int(round(float(task[4:]))))
            return [vocab_to_int["add"]] + [vocab_to_int[x] for x in val]
        elif task[:4] == "mult":
            val = str(int(round(float(task[5:]))))
            return [vocab_to_int["mult"]] + [vocab_to_int[x] for x in val]
        elif task[:7] == "permute":
            val = task[8:]
            return [vocab_to_int["permute"]] + [vocab_to_int[x] for x in val]
        elif task[:3] == "is_":
            if task[3] == "X":
                return [vocab_to_int[x] for x in task.split('_')] 
            else:
                return [vocab_to_int["is"], vocab_to_int[task[3:]]]
        else:
            raise ValueError("Unrecognized meta task: %s" % task)


## running stuff
for run_i in range(run_config["num_runs"]):
    np.random.seed(run_i)
    tf.set_random_seed(run_i)
    run_config["this_run"] = run_i

    model = poly_HoMM_model(run_config=run_config)
    model.run_training()

    tf.reset_default_graph()
