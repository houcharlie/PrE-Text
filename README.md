# PrE-Text

## Dataset processing
In the data folder you will need to split the {dataset}_train.json into datasets
of split into dictionaries of 1250 clients each with 8 text samples each in order,
and name it {dataset}_train_fed_1250.json. 

Also get the first 100000 samples from c4-en on huggingface and call this dataset initialization.json.

## Running the FL baselines
Use the base_config.json and populate the relevant fields into the json. Then you can run the baseline with

python run_fl_baseline.py --config-file {config_file}

## Running PrE-Text
First, run (edit the paths that have been commented out for the datasets in main.py)

python main.py -datadir {dataset} -num_clients {num_clients} -mask {masking level} -lookahead {number of lookahead} -multiplier {N_syn is this times 256} -embed 5 -t_steps {this is W_steps} -H_multiplier {H will be this times 4.0 * sensitivity} -trial {for mulitple trials}

Then for the expansion part (can adjust the code for the number of samples, set output paths in the code), 

python llama_bootstrap.py -datadir {dataset} -num_clients {num_clients} -mask {masking level} -lookahead {number of lookahead} -multiplier {N_syn is this times 256} -embed 5 -t_steps {this is W_steps} -H_multiplier {H will be this times 4.0 * sensitivity} -trial {for mulitple trials}

## Running the evals

For the DistilGPT2 eval, we can run the following (adjust paths in code)

python eval_gpt2.py -datadir {dataset} -num_clients {num_clients} -mask {masking level} -lookahead {number of lookahead} -multiplier {N_syn is this times 256} -embed 5 -t_steps {this is W_steps} -H_multiplier {H will be this times 4.0 * sensitivity} -trial {for mulitple trials}


For the LLaMA-2-7B eval, we can run the following (adjust paths in code)

python eval_llama2.py -datadir {dataset} -num_clients {num_clients} -mask {masking level} -lookahead {number of lookahead} -multiplier {N_syn is this times 256} -embed 5 -t_steps {this is W_steps} -H_multiplier {H will be this times 4.0 * sensitivity} -trial {for mulitple trials}
