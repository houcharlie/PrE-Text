import json
import os
import random
import sys
import time
import argparse


import numpy as np
import torch
from accelerate import Accelerator
from nn_histogram import NN_Histogram
from sentence_transformers import SentenceTransformer, models
from transformers import RobertaForMaskedLM, RobertaTokenizer
from variation import Variation
from similarity import Similarity
from opacus.accountants.analysis import rdp as privacy_analysis

# pyre-ignore[C901]
def main(args, accelerator):
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large", use_fast=True, cache_dir=args.cachedir)
    model = RobertaForMaskedLM.from_pretrained("roberta-large", torch_dtype=torch.float16, cache_dir=args.cachedir)
    accelerator.print('miniLM', file=sys.stderr)
    mpnet = SentenceTransformer("all-MiniLM-L6-v2", cache_folder = args.cachedir)
    

    
    datadir = f'./data/{args.datadir}_train.json'
    outputdir = args.outputdir
    seq_len = args.seq_len
    with open(datadir, encoding='utf8') as file:
        private_samples = json.load(file)

    accelerator.print(
        "Num private train samples", len(private_samples), file=sys.stderr
    )
    accelerator.print("Private samples", private_samples[:5], file=sys.stderr)
    
    
    scale = args.sensitivity * args.sigma
    rdp = privacy_analysis.compute_rdp(
        q=1.0,
        noise_multiplier=args.sigma,
        steps=11,
        orders=[1.0 + 0.1 * t for t in range(1, 1000)],
    )
    eps, opt_alpha = privacy_analysis.get_privacy_spent(
            orders=[1.0 + 0.1 * t for t in range(1, 1000)], rdp=rdp, delta=args.delta
        )
    accelerator.print("Epsilon of this run", eps, file=sys.stderr)
    
    model = model.eval()
    model = accelerator.prepare_model(model, evaluation_mode=True)

    config = {
        "model": model,
        "tokenizer": tokenizer,
        "accelerator": accelerator,
        "batch_size": 256,
        "max_length": seq_len,
        "num_workers": 1,
        "num_gpus": 1,
        "embed_batch_size": 512,
        "mpnet": mpnet,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "nearest_neighbors_print": 3,
        "sigma": scale * 1.541 * np.sqrt(2), # formerly 1.541 sqrt(2)
        "H": scale * 4.0 * args.H_multiplier, # formerly 4.0
        "embed_dim": mpnet.get_sentence_embedding_dimension(),
        "lookahead": args.lookahead,
        "T": 11,
        "multiplier": args.multiplier,
        "device": 'cuda',
        "t_steps": args.t_steps,
    }
    config["nsyn"] = config["batch_size"] * config["multiplier"] * config['num_gpus']
    output_dir = os.path.join(outputdir,
        "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}/".format(
            args.datadir,
            args.mask,
            config["lookahead"],
            config["nsyn"],
            args.t_steps,
            args.H_multiplier,
            args.sensitivity,
            args.sigma,
            args.delta,
            args.trial
        )
    )
    accelerator.print(output_dir, file=sys.stderr)
   
    with open('./data/initialization.json', encoding='utf8') as f:
        load_list = json.load(f)
                
    load_list = [x for x in load_list if len(x.split(" ")) > 20]
    init_pop = load_list

    if accelerator.is_main_process:
        accelerator.print("Initial population size", len(load_list), file=sys.stderr)
    accelerator.print('Init pop size', len(init_pop))
    schedule = [args.mask for i in range(config["T"])]
    accelerator.print(output_dir, file=sys.stderr)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if accelerator.is_main_process:
        accelerator.print(init_pop[0], file=sys.stderr)
        accelerator.print("Schedule", schedule, file=sys.stderr)

    parent_texts = random.choices(init_pop, k=config["nsyn"])
    parent_texts = sorted(parent_texts, key=lambda x: len(x))
    accelerator.print("Num parent texts", len(parent_texts), file=sys.stderr)
    parent_set = tokenizer(
        parent_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config["max_length"],
    )
    num_steps_current = 0
    dist_list = []

    if not os.path.isfile(os.path.join(output_dir, 'private_embeds.np')):
        accelerator.print("Making private embeddings", file=sys.stderr)
        t0 = time.time()
        private_embeddings = Similarity.concat_embedding(
            private_samples, config
        )
        t1 = time.time()
        accelerator.print("Time for private embeddings", t1 - t0, file=sys.stderr)
        np.save(os.path.join(output_dir, 'private_embeds.np'), private_embeddings)
    else:
        private_embeddings = np.load(os.path.join(output_dir, 'private_embeds.np'))
        accelerator.print('Embeddings shape', private_embeddings.shape)

    for t in range(config["T"]):
        attention_mask_pad_sums = torch.sum(parent_set['attention_mask'] == 0, axis=1)
        curr_inputs_pad_sums = torch.sum(parent_set['input_ids'] == 1, axis=1)
        all_ok = torch.sum(curr_inputs_pad_sums - attention_mask_pad_sums) == 0
        accelerator.print('At top all_ok?', all_ok)
        t0 = time.time()

        histogram, meandist, nearest_idx = NN_Histogram.dp_nn_histogram(
            private_embeddings,
            parent_set["input_ids"],
            parent_set["attention_mask"],
            schedule[t],
            config,
        )
        dist_list.append(meandist)
        accelerator.print('Current step', t, file=sys.stderr)
        accelerator.print('Dist list', dist_list, file=sys.stderr)
        accelerator.print(
            "Nearest generated samples",
            [
                tokenizer.batch_decode(
                    [parent_set["input_ids"][idx, :]], skip_special_tokens=True
                )
                for idx in nearest_idx
            ],
        )
        accelerator.print("Mean dist from nearest neighbor", meandist, file=sys.stderr)

        accelerator.print("Histogram sum", np.sum(histogram), file=sys.stderr)
        t1 = time.time()
        if accelerator.is_main_process:
            accelerator.print("Histogram time:", t1 - t0, file=sys.stderr)
            accelerator.print("Producing surviving parents...", file=sys.stderr)
        t0 = time.time()

        indices = np.random.choice(
            config["nsyn"], config["nsyn"], p=histogram / np.sum(histogram)
        )
        indices = np.sort(indices)
        surviving_parents_ids = parent_set["input_ids"][indices, :]
        surviving_parents_mask = parent_set["attention_mask"][indices, :]

        attention_mask_pad_sums = torch.sum(surviving_parents_mask == 0, axis=1)
        curr_inputs_pad_sums = torch.sum(surviving_parents_ids == 1, axis=1)
        all_ok = torch.sum(curr_inputs_pad_sums - attention_mask_pad_sums) == 0
        accelerator.print('After sampling all ok?', all_ok)
        t1 = time.time()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.print("Choosing survivors time:", t1 - t0, file=sys.stderr)
            accelerator.print("Producing variations...", file=sys.stderr)
        t0 = time.time()
        new_variations = Variation.produce_variation(
            {
                "input_ids": surviving_parents_ids,
                "attention_mask": surviving_parents_mask,
            },
            schedule[t],
            config,
        )

        attention_mask_pad_sums = torch.sum(new_variations['attention_mask']== 0, axis=1)
        curr_inputs_pad_sums = torch.sum(new_variations['input_ids'] == 1, axis=1)
        all_ok = torch.sum(curr_inputs_pad_sums - attention_mask_pad_sums) == 0
        accelerator.print('Produced variations all_ok?', all_ok)

        t1 = time.time()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.print("Producing variations time", t1 - t0, file=sys.stderr)
            accelerator.print("Checking similarity...", file=sys.stderr)

        generated_samples = tokenizer.batch_decode(
            new_variations['input_ids'], skip_special_tokens=True
        )
        surviving_samples = tokenizer.batch_decode(
            surviving_parents_ids, skip_special_tokens=True
        )

        parent_set["input_ids"] = new_variations['input_ids']
        parent_set["attention_mask"] = new_variations['attention_mask']

        num_steps_current += 1
        if accelerator.is_main_process:
            text_list = generated_samples
            with open(
                os.path.join(output_dir, f"generated_text_it{t}.json"),
                "w+",
                encoding="utf8",
            ) as json_file:
                json.dump(text_list, json_file, ensure_ascii=False)
            with open(
                os.path.join(output_dir, f"surviving_text_it{t}.json"),
                "w+",
                encoding="utf8",
            ) as json_file:
                json.dump(list(set(surviving_samples)), json_file, ensure_ascii=False)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    accelerator = Accelerator()
    parser = argparse.ArgumentParser('Run PE-text.')
    parser.add_argument('-datadir', type=str, default='')
    parser.add_argument('-outputdir', type=str, required=True)
    parser.add_argument('-cachedir', type=str, required=True)
    parser.add_argument('-sensitivity', type=int, required=True)
    parser.add_argument('-delta', type=float, default=3e-6)
    parser.add_argument('-sigma', type=float, required=True)
    parser.add_argument('-mask', type=float, default=0.3)
    parser.add_argument('-lookahead', type=int, default=4)
    parser.add_argument('-multiplier', type=int, default=4)
    parser.add_argument('-seq_len', type=int, default=64)
    parser.add_argument('-t_steps', type=int, default=2)
    parser.add_argument('-trial', type=int, default=0)
    parser.add_argument('-H_multiplier', type=float, default=0.25)
    with accelerator.main_process_first():
        argobj = parser.parse_args()
    print(argobj)
    main(argobj, accelerator)