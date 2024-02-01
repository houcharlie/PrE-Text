# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import os
from accelerate import Accelerator
from sentence_transformers import models, SentenceTransformer
from transformers import RobertaForMaskedLM, RobertaTokenizer
from variation import Variation

class Similarity:
    @staticmethod
    def sentence_embedding(texts, embedding_model, device='cuda'):
        """
        texts: list of texts
        config: holds all the various things you need
        returns: sentence embeddings
        """
        sentence_embeddings = embedding_model.encode(
            texts, device=device
        )
        return np.vstack(sentence_embeddings)


    @staticmethod
    def concat_embedding(texts, config):
        mpnet_embeds = Similarity.sentence_embedding(texts, config['mpnet'], device=config['device'])
        if config['bart'] is not None:
            bart_embeds = Similarity.sentence_embedding(texts, config['bart'], device=config['device'])
            return np.concatenate((4.0 * mpnet_embeds, bart_embeds), axis=1)
        else:
            return mpnet_embeds



    @staticmethod
    def lookahead_embedding(parent_set, attention_mask, mlm_probability, config):
        """
        texts: list of texts:
        config: holds the configurations
        returns: sentence_embeddings
        """
        tokenizer = config["tokenizer"]
        embeddings_list = []
        for _ in range(config["lookahead"]):
            curr_variation = Variation.produce_variation(
                {"input_ids": parent_set, "attention_mask": attention_mask},
                mlm_probability,
                config,
            )['input_ids']
            curr_variation_texts = tokenizer.batch_decode(
                curr_variation, ignore_special_tokens=True
            )
            curr_variation_embedding = Similarity.concat_embedding(
                curr_variation_texts, config
            )[None, :, :]
            embeddings_list.append(curr_variation_embedding)
        embeddings_cat = np.concatenate(embeddings_list, axis=0)
        embeddings_mean = np.mean(embeddings_cat, axis=0)
        return embeddings_mean


if __name__ == "__main__":
    population = [
        "The capital of France is Paris.",
        "The capital of France is Paris.",
        "The capital of France is Paris.",
        "The capital of France is Paris.",
        "The meaning of life is 42.",
        "The meaning of life is 42.",
        "The meaning of life is 42.",
        "The meaning of life is 42.",
    ] * 100000
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large", use_fast=True)
    model = RobertaForMaskedLM.from_pretrained(
        "roberta-large",
        torch_dtype=torch.float16
    )
    accelerator = Accelerator()
    model = model.eval()
    model = accelerator.prepare(model)
    mpnet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder='/ocean/projects/cis230033p/houc/LLM/pretrained_models')
    bart = models.Transformer("facebook/bart-large", cache_dir='/ocean/projects/cis230033p/houc/LLM/pretrained_models')
    pooling_model = models.Pooling(bart.get_word_embedding_dimension())
    bart = SentenceTransformer(modules=[bart, pooling_model])

    seq_len = 64

    config = {
        "model": model,
        "tokenizer": tokenizer,
        "accelerator": accelerator,
        "batch_size": 256,
        "max_length": seq_len,
        "num_workers": 8,
        "embed_batch_size": 32,
        "mpnet": mpnet,
        "bart": bart,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "embed_dim": mpnet.get_sentence_embedding_dimension() + bart.get_sentence_embedding_dimension(),
        "lookahead": 2,
    }
    with accelerator.main_process_first():
        population_tok = tokenizer(
            population,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=config["max_length"],
        )

    """
    This tests Similarity.sentence_embedding
    """
    sentence_embeds = Similarity.sentence_embedding(
        population, config["mpnet"]
    )
    accelerator.print(sentence_embeds.shape)
    if accelerator.is_main_process:
        print(
            "Norm between same items",
            np.linalg.norm(sentence_embeds[0, :] - sentence_embeds[2, :]),
        )
        print(
            "Norm between diff items",
            np.linalg.norm(sentence_embeds[0, :] - sentence_embeds[1, :]),
        )
        assert np.linalg.norm(
            sentence_embeds[0, :] - sentence_embeds[1, :]
        ) < np.linalg.norm(sentence_embeds[0, :] - sentence_embeds[4, :])
        print("Test 1 passed!")
    """
    This tests Similarity.concat
    """
    sentence_embeds = Similarity.concat_embedding(
        population, config
    )
    accelerator.print(sentence_embeds.shape)
    if accelerator.is_main_process:
        print(
            "Norm between same items",
            np.linalg.norm(sentence_embeds[0, :] - sentence_embeds[2, :]),
        )
        print(
            "Norm between diff items",
            np.linalg.norm(sentence_embeds[0, :] - sentence_embeds[1, :]),
        )
        assert np.linalg.norm(
            sentence_embeds[0, :] - sentence_embeds[1, :]
        ) < np.linalg.norm(sentence_embeds[0, :] - sentence_embeds[4, :])
        print("Test 1 passed!")

    """
    This tests Similarity.lookahead_embedding
    """

    sentence_embeds = Similarity.lookahead_embedding(
        population_tok["input_ids"], population_tok["attention_mask"], 0.3, config
    )
    if accelerator.is_main_process:
        print(
            "Norm between same items",
            np.linalg.norm(sentence_embeds[0, :] - sentence_embeds[1, :]),
        )
        print(
            "Norm between diff items",
            np.linalg.norm(sentence_embeds[0, :] - sentence_embeds[4, :]),
        )
        assert np.linalg.norm(
            sentence_embeds[0, :] - sentence_embeds[1, :]
        ) < np.linalg.norm(sentence_embeds[0, :] - sentence_embeds[4, :])
        print("Test 2 passed!")

