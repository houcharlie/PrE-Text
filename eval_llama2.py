import argparse
import os
import json
import transformers
import re
from datasets import Dataset, load_metric
from transformers import AutoModelForCausalLM, LlamaTokenizer
from copy import deepcopy
import numpy as np
import torch
import random
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Run expansion.')
    parser.add_argument('-datadir', type=str, default='linkedin')
    parser.add_argument('-num_clients', type=int, default=2500)
    parser.add_argument('-mask', type=float, default=0.3)
    parser.add_argument('-lookahead', type=int, default=4)
    parser.add_argument('-multiplier', type=int, default=1)
    parser.add_argument('-embed', type=int, default=1)
    parser.add_argument('-t_steps', type=int, default=3)
    parser.add_argument('-H_multiplier', type=float, default=1.0)
    parser.add_argument('-trial', type=int, default=0)
    # parser.add_argument('-machine_num', type=int, default=0)
    args = parser.parse_args()
    outputdir = ''
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
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir='')
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir='', load_in_8bit=True, torch_dtype=torch.float16)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=['q_proj', 'o_proj', 'v_proj', 'k_proj', 'gate_proj', 'up_proj', 'down_proj', 'lm_head'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    all_data = []
    for i in range(1):
        # with open(os.path.join(output_dir, f'surviving_text_it{i}.json'), 'r', encoding='utf8') as f:
        with open(os.path.join(output_dir, f'llama7b_text_syn_{i}.json'), 'r', encoding='utf8') as f:
            s = json.load(f)
        for text in s:
            split_samples = re.split("\n\nOriginal", text)
            if len(split_samples) > 0:
                if len(split_samples[0].split(' ')) > 3:
                    all_data.append(split_samples[0].replace('\n\n', ' ').replace('\n', ' '))
    all_data = random.sample(all_data, 100000)
    print('Length of train data', len(all_data))
    syn_data_dict = [{'text': x} for x in all_data]
    cutoff_len = 32
    syn_data = Dataset.from_list(syn_data_dict)
    with open(os.path.join(f'{args.datadir}_eval.json'), 'r', encoding='utf8') as f:
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

    syn_data_tokenized = syn_data.shuffle().map(tokenize, num_proc=5)
    test_data_tokenized = test_data.shuffle().map(tokenize, num_proc=5)
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
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            gradient_accumulation_steps=16 * 4,
            # auto_find_batch_size=True,
            lr_scheduler_type='constant',
            num_train_epochs=1,
            disable_tqdm=True,
            learning_rate=5e-5,
            report_to='none',
            eval_steps=1,
            logging_steps=1,
            save_steps=1000000,
            ddp_find_unused_parameters=False,
            evaluation_strategy="epoch",
            logging_strategy="steps",
            optim="adamw_torch",
            output_dir=os.path.join(output_dir, 'transformers_llama_outputs')
        ),
        # compute_metrics=compute_metrics
        # data_collator=DefaultDataCollator,
    )
    
    trainer.train()
    best_metrics = trainer.evaluate()
    with open(os.path.join(output_dir, 'evaluation_llama2_metrics.json'), 'w+') as f:
        json.dump(best_metrics, f)