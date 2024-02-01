import sys
import time

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import numpy as np
import torch
from accelerate import Accelerator
from sentence_transformers import models, SentenceTransformer
from similarity import Similarity
from transformers import RobertaForMaskedLM, RobertaTokenizer


class NN_Histogram:
    @staticmethod
    def split_given_size(a, size):
        return np.split(a, np.arange(size,len(a),size))
    @staticmethod
    def dp_nn_histogram(
        private_embeddings, parent_set, attention_mask, mlm_probability, config
    ):
        """
        private_samples: list of private texts
        synthetic_samples: list of synthetic texts
        config: config dict
        """
        sigma = config["sigma"]
        H = config["H"]
        embed_dim = config["embed_dim"]
        accelerator = config["accelerator"]
        nearest_neighbors_print = config["nearest_neighbors_print"]
        
        t0 = time.time()
        lookahead_embeddings = Similarity.lookahead_embedding(parent_set, attention_mask, 
                                                              mlm_probability, config)
        
        t1 = time.time()
        accelerator.print("Time for synthetic embeddings:", t1 - t0, file=sys.stderr)
        index_flat = faiss.IndexFlatL2(embed_dim)
        index_flat.add(lookahead_embeddings)
        t0 = time.time()
        
        # 29056399 private samples
        # batch into batches of 100000
        private_embeds_split = NN_Histogram.split_given_size(private_embeddings, 50000)
        res_D = []
        res_I = []
        for curr_idx, curr_embedding in enumerate(private_embeds_split):
            # split into 8 items, one for each process
            len_curr_batch = curr_embedding.shape[0]
            closest_multiple = (len_curr_batch // config['num_workers']) * config['num_workers']
            randidx = np.random.choice(len_curr_batch, closest_multiple, replace=False)
            curr_embedding_subset = curr_embedding[randidx,:]
            process_split = np.split(curr_embedding_subset, config['num_workers'], axis=0)
            with accelerator.split_between_processes(process_split) as private_search_embed:
                D, I = index_flat.search(private_search_embed[0], 1)
                D = torch.from_numpy(D).to('cuda:{0}'.format(accelerator.process_index))
                I = torch.from_numpy(I).to('cuda:{0}'.format(accelerator.process_index))
            D = accelerator.gather(D).cpu().numpy()
            I = accelerator.gather(I).cpu().numpy()
            res_D.append(D)
            res_I.append(I)
        I = np.concatenate(res_I, axis=0)
        D = np.concatenate(res_D, axis=0)
        # n_priv x 1
        accelerator.print(I.shape, file=sys.stderr)
        resulting_histogram, _ = np.histogram(
            I[:, 0],
            bins=[-0.5] + [x + 0.5 for x in range(lookahead_embeddings.shape[0])],
        )
        resulting_mean_distance = np.mean(D, axis=0).squeeze()
        first_nearest_neighbors_idx = I[:nearest_neighbors_print, 0]
        resulting_histogram_noised = (
            resulting_histogram.astype(np.float32)
            + np.random.standard_normal(size=resulting_histogram.shape) * sigma
        )
        accelerator.print('Histogram before thresholding', np.sum(np.maximum(resulting_histogram_noised, 0)))
        noised_histogram_thresh = np.maximum(resulting_histogram_noised - H, 0.0)
        accelerator.print('Histogram after thresholding', np.sum(noised_histogram_thresh))
        t1 = time.time()
        accelerator.print("Time for nearest neighbor calculation:", t1 - t0, file=sys.stderr)
        return (
            noised_histogram_thresh,
            resulting_mean_distance,
            first_nearest_neighbors_idx,
        )


"""
This unit test tests whether the returned histogram in NN_histogram.dp_nn_histogram() satisfies sanity checks.
"""
if __name__ == "__main__":
    population = ["Lol married different", "Yeah but not that far in advance", "The capital of France is Paris.", "The meaning of life is 42."]
    private_samples = [
        "The capital of France is Paris.",
        "The meaning of life is 42.",
    ]
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large", use_fast=True)
    model = RobertaForMaskedLM.from_pretrained(
        "roberta-large",
        torch_dtype=torch.float16
    )
    accelerator = Accelerator()
    model = model.eval()
    model = accelerator.prepare_model(model, evaluation_mode=True)

    mpnet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder='/ocean/projects/cis230033p/houc/LLM/pretrained_models')
    bart = models.Transformer("facebook/bart-large", cache_dir='/ocean/projects/cis230033p/houc/LLM/pretrained_models')
    pooling_model = models.Pooling(bart.get_word_embedding_dimension())
    bart = SentenceTransformer(modules=[bart, pooling_model])
    seq_len = 32

    config = {
        "model": model,
        "tokenizer": tokenizer,
        "accelerator": accelerator,
        "batch_size": 1,
        "max_length": seq_len,
        "num_workers": 8,
        "embed_batch_size": 32,
        "bart": bart,
        "mpnet": mpnet,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "embed_dim": bart.get_sentence_embedding_dimension() + mpnet.get_sentence_embedding_dimension(),
        "sigma": 0.0,
        "H": 0.0,
        "nearest_neighbors_print": 3,
        "lookahead": 2,
    }
    model = accelerator.prepare(model)

    population_inputs = tokenizer(
        population, return_tensors="pt", truncation=True, padding=True, max_length=20
    )
    curr_hist, mean_dist, nearest_idx = NN_Histogram.dp_nn_histogram(
        private_samples,
        population_inputs["input_ids"],
        population_inputs["attention_mask"],
        0.2,
        config,
    )
    accelerator.print(curr_hist)
    accelerator.print(mean_dist)
    assert np.linalg.norm(curr_hist - np.array([0., 0., 1.0, 1.0])) < 1e-10
    accelerator.print("Test passed!")