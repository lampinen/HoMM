from itertools import permutations
from copy import deepcopy

import ..HoMM 
from ..configs.default_run_config import default_run_config
import polynomials

run_config = default_run_config
run_config.update({
    "output_dir": "polynomials_results/",
    
    "memory_buffer_size": 1024, # How many points for each polynomial are stored
    "meta_batch_size": 50, # how many meta-learner sees
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

class poly_HoMM_model(HoMM.HoMM_model):
    def __init__(self, run_config=None):
        super(poly_HoMM_model, self).__init__(run_config=run_config)

    def _pre_build_calls(self):
        # set up the base tasks
        poly_fam = polynomials.polynomial_family(config["num_variables"], config["max_degree"])
        self.run_config["variables"] = poly_fam.variables

        self.base_train = [poly_fam.sample_polynomial(coefficient_sd=config["poly_coeff_sd"]) for _ in range(self.run_config["num_base_train_tasks"])]
        self.base_eval = [poly_fam.sample_polynomial(coefficient_sd=config["poly_coeff_sd"]) for _ in range(self.run_config["num_base_eval_tasks"])]

        # set up the meta tasks

        self.meta_class_train = ["is_constant_polynomial"] + ["is_intercept_nonzero"] + ["is_%s_relevant" % var for var in self.run_config["variables"]]
        self.meta_class_eval = [] 

        self.meta_map_train = ["square"] + ["add_%f" % c for c in self.run_config["meta_add_vals"]] + ["mult_%f" % c for c in self.run_config["meta_mult_vals"]]
        self.meta_map_eval = self.run_config["new_meta_mappings"] 

        permutation_mappings = ["permute_" + "".join([str(x) for x in p]) for p in permutations(range(self.run_config["num_variables"]))]
        np.random.seed(0)
        np.random.shuffle(permutation_mappings)
        self.meta_map_train += permutation_mappings[:len(permutation_mappings)//2]
        self.meta_map_eval += permutation_mappings[len(permutation_mappings)//2:]

        # set up language

        #vocab = ['PAD'] + [str(x) for x in range(10)] + [".", "+", "-", "^"] + poly_fam.variables
        #vocab_to_int = dict(zip(vocab, range(len(vocab))))
        #
        #self.run_config["vocab"] = vocab

        # set up the meta pairings 
        self.meta_pairings, implied_tasks_train, implied_tasks_eval = polynomials.get_meta_pairings(
            base_train=self.run_config["base_train"],
            base_eval=self.run_config["base_eval"],
            meta_class_train=self.run_config["meta_class_train"],
            meta_class_eval=self.run_config["meta_class_eval"],
            meta_map_train=self.run_config["meta_map_train"],
            meta_map_eval=self.run_config["meta_map_eval"]) 

        # add the base tasks implied by the mappings
        self.run_config["base_train"] += implied_tasks_train
        self.run_config["base_eval"] += implied_tasks_eval


    def fill_buffers(self, num_data_points=1, include_new=False):
        """Add new "experiences" to memory buffers."""
        this_tasks = 
        for t in this_tasks:
            buff = self.memory_buffers[polynomials.stringify_polynomial(t)]
            x_data = np.zeros([num_data_points, self.config["num_input"]])
            y_data = np.zeros([num_data_points, self.config["num_output"]])
            for point_i in range(num_data_points):
                point = t.family.sample_point(val_range=self.config["point_val_range"])
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
