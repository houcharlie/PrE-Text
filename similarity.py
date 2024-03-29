
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
