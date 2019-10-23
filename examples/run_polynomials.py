from polynomials import *
from itertools import permutations

import ..HoMM 
from ..configs.default_run_config import default_run_config

run_config = default_run_config.update({
    "output_dir": "results/",
    
    "memory_buffer_size": 1024, # How many points for each polynomial are stored
    "meta_batch_size": 50, # how many meta-learner sees
    "num_base_tasks": 60, # prior to meta-augmentation

    "num_variables": 4,
    "max_degree": 2,
    "num_new_tasks": 40,
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
        self.poly_fam = polynomial_family(config["num_variables"], config["max_degree"])
        self.run_config["variables"] = poly_fam.variables

        self.run_config["meta_class_train"] = ["is_constant_polynomial"] + ["is_intercept_nonzero"] + ["is_%s_relevant" % var for var in self.run_config["variables"]]

        self.run_config["base_meta_mappings"] = ["square"] + ["add_%f" % c for c in self.run_config["meta_add_vals"]] + ["mult_%f" % c for c in self.run_config["meta_mult_vals"]]
        permutation_mappings = ["permute_" + "".join([str(x) for x in p]) for p in permutations(range(self.run_config["num_variables"]))]
        np.random.seed(0)
        np.random.shuffle(permutation_mappings)
        self.run_config["train_meta_mappings"] += permutation_mappings[:len(permutation_mappings)//2]
        self.run_config["new_meta_mappings"] += permutation_mappings[len(permutation_mappings)//2:]

        self.run_config["base_meta_tasks"] = [x for x in self.run_config["base_meta_tasks"] if x not in self.run_config["new_meta_tasks"]]
        self.run_config["meta_mappings"] = [x for x in self.run_config["base_meta_mappings"] if x not in self.run_config["new_meta_mappings"]]

        # set up language

        #vocab = ['PAD'] + [str(x) for x in range(10)] + [".", "+", "-", "^"] + poly_fam.variables
        #vocab_to_int = dict(zip(vocab, range(len(vocab))))
        #
        #self.run_config["vocab"] = vocab

        # set up the meta_task generator


    def fill_buffers(self, num_data_points=1, include_new=False):
        """Add new "experiences" to memory buffers."""
        this_tasks = 
        for t in this_tasks:
            buff = self.memory_buffers[_stringify_polynomial(t)]
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

    def run_training(self, filename_prefix, num_epochs, include_new=False):
        """Train model on base and meta tasks, if include_new include also
        the new ones."""
        config = self.config
        loss_filename = filename_prefix + "_losses.csv"
        sweep_filename = filename_prefix + "_sweep_losses.csv"
        meta_filename = filename_prefix + "_meta_true_losses.csv"
        lang_filename = filename_prefix + "_language_losses.csv"
        train_language = config["train_language"]
        train_base = config["train_base"]
        train_meta = config["train_meta"]

        with open(loss_filename, "w") as fout, open(meta_filename, "w") as fout_meta, open(lang_filename, "w") as fout_lang:
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

            if train_language:
                (base_lang_names, 
                 base_lang_losses) = self.run_base_language_eval(
                    include_new=include_new)
                lang_loss_format = ", ".join(["%f" for _ in base_lang_names]) + "\n"
                fout_lang.write("epoch, " + ", ".join(base_lang_names) + "\n")

            s_epoch  = "0, "
            curr_losses = s_epoch + (loss_format % tuple(
                base_losses + meta_losses))
            curr_meta_true = s_epoch + (meta_true_format % tuple(
                meta_true_losses))
            fout.write(curr_losses)
            fout_meta.write(curr_meta_true)

            if train_language:
                curr_lang_losses = s_epoch + (lang_loss_format % tuple(
                    base_lang_losses))
                fout_lang.write(curr_lang_losses)

            if config["sweep_meta_batch_sizes"] is not None:
                with open(sweep_filename, "w") as fout_sweep:
                    sweep_names, sweep_losses = self.run_base_eval(
                        include_new=include_new, sweep_meta_batch_sizes=config["sweep_meta_batch_sizes"])
                    fout_sweep.write("epoch, size, " + ", ".join(base_names) + "\n")
                    for i, swept_batch_size in enumerate(config["sweep_meta_batch_sizes"]):
                        swept_losses = s_epoch + ("%i, " % swept_batch_size) + (base_loss_format % tuple(sweep_losses[i]))
                        fout_sweep.write(swept_losses)

            if include_new:
                tasks = self.all_tasks_with_implied
                learning_rate = config["new_init_learning_rate"]
                language_learning_rate = config["new_init_language_learning_rate"]
                meta_learning_rate = config["new_init_meta_learning_rate"]
            else:
                tasks = self.all_initial_tasks_with_implied
                learning_rate = config["init_learning_rate"]
                language_learning_rate = config["init_language_learning_rate"]
                meta_learning_rate = config["init_meta_learning_rate"]

            save_every = config["save_every"]
            early_stopping_thresh = config["early_stopping_thresh"]
            lr_decays_every = config["lr_decays_every"]
            lr_decay = config["lr_decay"]
            language_lr_decay = config["language_lr_decay"]
            meta_lr_decay = config["meta_lr_decay"]
            min_learning_rate = config["min_learning_rate"]
            min_meta_learning_rate = config["min_meta_learning_rate"]
            min_language_learning_rate = config["min_language_learning_rate"]


            self.fill_buffers(num_data_points=config["memory_buffer_size"],
                              include_new=True)
            self.refresh_meta_dataset_cache(include_new=include_new)
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
                        if train_meta:
                            dataset = self.meta_dataset_cache[task]
                            self.meta_train_step(dataset, meta_learning_rate)
                    else:
                        str_task = _stringify_polynomial(task)
                        memory_buffer = self.memory_buffers[str_task]
                        if train_base:
                            self.base_train_step(memory_buffer, learning_rate)
                        if train_language:
                            intified_task = self.task_to_ints[str_task]
                            if intified_task is not None:
                                self.base_language_train_step(
                                    intified_task, memory_buffer,
                                    language_learning_rate)


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
                    if train_language:
                        (_, base_lang_losses) = self.run_base_language_eval(
                            include_new=include_new)
                        curr_lang_losses = s_epoch + (lang_loss_format % tuple(
                            base_lang_losses))
                        fout_lang.write(curr_lang_losses)
                        print(curr_losses, curr_lang_losses)
                        if np.all(curr_losses < early_stopping_thresh) and np.all(curr_lang_losses < early_stopping_thresh):
                            print("Early stop!")
                            break
                    else:
                        print(curr_losses)
                        if np.all(curr_losses < early_stopping_thresh):
                            print("Early stop!")
                            break


                if epoch % lr_decays_every == 0 and epoch > 0:
                    if learning_rate > min_learning_rate:
                        learning_rate *= lr_decay

                    if meta_learning_rate > min_meta_learning_rate:
                        meta_learning_rate *= meta_lr_decay

                    if train_language and language_learning_rate > min_language_learning_rate:
                        language_learning_rate *= language_lr_decay


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
    config["base_tasks"] = [poly_fam.sample_polynomial(coefficient_sd=config["poly_coeff_sd"]) for _ in range(config["num_base_tasks"])]
    config["new_tasks"] = [poly_fam.sample_polynomial(coefficient_sd=config["poly_coeff_sd"]) for _ in range(config["num_new_tasks"])]
    config["base_task_names"] = [x.to_symbols() for x in config["base_tasks"]]
    config["new_task_names"] = [x.to_symbols() for x in config["new_tasks"]]

# tasks implied by meta mappings, network will also be trained on these
    config["implied_base_tasks"] = []
    config["implied_new_tasks"] = []

    filename_prefix = config["output_dir"] + "run%i" % run_i
    print("Now running %s" % filename_prefix)
    _save_config(filename_prefix + "_config.csv", config)


    model = poly_HoMM_model(config)
#    model.save_embeddings(filename=filename_prefix + "_init_embeddings.csv",
#                          include_new=False)
    if config["restore_checkpoint_path"] is not None:
        model.restore_parameters(config["restore_checkpoint_path"] + "run%i" % run_i + "_guess_checkpoint")
    else:
        model.run_training(filename_prefix=filename_prefix,
                           num_epochs=config["max_base_epochs"],
                           include_new=False)
#    cProfile.run('model.run_training(filename_prefix=filename_prefix, num_epochs=config["max_base_epochs"], include_new=False)')

    model.save_parameters(filename_prefix + "_guess_checkpoint")
#    model.save_embeddings(filename=filename_prefix + "_guess_embeddings.csv",
#                          include_new=True)

    model.run_training(filename_prefix=filename_prefix + "_new",
                       num_epochs=config["max_new_epochs"],
                       include_new=True)
#    cProfile.run('model.run_training(filename_prefix=filename_prefix + "_new", num_epochs=config["max_new_epochs"], include_new=True)')
    model.save_parameters(filename_prefix + "_final_checkpoint")

#    model.save_embeddings(filename=filename_prefix + "_final_embeddings.csv",
#                          include_new=True)

    tf.reset_default_graph()
