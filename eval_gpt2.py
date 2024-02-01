import argparse
import os
import json
import transformers
import re
from datasets import Dataset, load_metric
from transformers import AutoModelForCausalLM, GPT2Tokenizer, TrainerCallback
from copy import deepcopy
import numpy as np
import torch
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Run expansion.')
    parser.add_argument('-datadir', type=str, default='linkedin')
    parser.add_argument('-num_clients', type=int, default=2500)
    parser.add_argument('-mask', type=float, default=0.3)
    parser.add_argument('-lookahead', type=int, default=4)
    parser.add_argument('-multiplier', type=int, default=1)
    parser.add_argument('-embed', type=int, default=1)
    parser.add_argument('-t_steps', type=int, default=3)
    parser.add_argument('-trial', type=int, default=0)
    parser.add_argument('-H_multiplier', type=float, default=1.0)
    # parser.add_argument('-machine_num', type=int, default=0)
    args = parser.parse_args()
    outputdir = '/ocean/projects/cis230033p/houc/LLM/PE-results-noshuffle'
    output_dir = os.path.join(outputdir,
        "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}/".format(
            args.datadir,
            args.num_clients,
            args.mask,
            args.lookahead,
            args.multiplier * 256,
            args.embed,
            args.t_steps,
            args.H_multiplier,
            args.trial
        )
    )
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2", cache_dir='/ocean/projects/cis230033p/houc/LLM/pretrained_models')
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", cache_dir='/ocean/projects/cis230033p/houc/LLM/pretrained_models')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    all_data = []
    for i in range(1):
        with open(os.path.join(output_dir, f'llama7b_text_syn_{i}.json'), 'r', encoding='utf8') as f:
            s = json.load(f)
        for text in s:
            split_samples = re.split("Orig", text)
            raw_sample = split_samples[0]
            raw_sample = raw_sample.strip()
            raw_sample = raw_sample.strip('\n')
            if len(raw_sample.split(' ')) > 3:
                all_data.append(raw_sample.replace('\n\n', ' ').replace('\n', ' '))
    # all_data = all_data[:10000]
    print('Length of train data', len(all_data))
    syn_data_dict = [{'text': x} for x in all_data]
    cutoff_len = 64
    syn_data = Dataset.from_list(syn_data_dict)
    with open(os.path.join(f'/ocean/projects/cis230033p/houc/LLM/raw_datasets/fed_data/data/{args.datadir}_eval.json'), 'r', encoding='utf8') as f:
        test_data_raw = json.load(f)["1"]

    test_data_dict = [{'text': x} for x in test_data_raw]
    test_data = Dataset.from_list(test_data_dict)

    def tokenize(example):
        result = tokenizer(
            example['text'],
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
            return_tensors=None
        )
        result["labels"] = [-100 if token == tokenizer.pad_token_id else token for token in result['input_ids']]
        return result

    syn_data_tokenized = syn_data.shuffle().map(tokenize)
    test_data_tokenized = test_data.shuffle().map(tokenize)
    metric = load_metric('accuracy')

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        labels = np.append(labels[:,1:], -100 * np.ones((labels.shape[0], 1)), axis=1)
        
        matches = predictions == labels
        total = labels != -100
        total_num = np.sum(total)
        correct = np.sum(matches)
        return {'eval_accuracy': correct/total_num}
    trainer = transformers.Trainer(
        model=model,
        train_dataset=syn_data_tokenized,
        eval_dataset=test_data_tokenized,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            gradient_accumulation_steps=512,
            # auto_find_batch_size=True,
            num_train_epochs=10,
            lr_scheduler_type='constant',
            disable_tqdm=False,
            learning_rate=0.00018,
            report_to='none',
            logging_steps=1,
            eval_steps=1,
            save_steps=1000000,
            ddp_find_unused_parameters=False,
            evaluation_strategy="epoch",
            logging_strategy="steps",
            optim="adamw_torch",
            output_dir=os.path.join(output_dir, 'transformers_outputs')
        ),
        # compute_metrics=compute_metrics
        # data_collator=DefaultDataCollator,
    )
    
    trainer.train()
    best_metrics = trainer.evaluate()
    with open(os.path.join(output_dir, 'evaluation_distilgpt_metrics.json'), 'w+') as f:
        json.dump(best_metrics, f)