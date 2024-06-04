import torch
import argparse
import datasets
import os
import json
import re
import torch.nn as nn
import sys
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm import tqdm
from datasets import Dataset
    

def evaluate(model, eval_loader, accelerator, xent_loss):
    model.eval()
    total_loss = 0.0
    top_k_accuracies = {1: 0, 3: 0, 5: 0, 10: 0, 50: 0, 100: 0}
    total_evaluated_tokens = 0

    with torch.no_grad():
        for batch in eval_loader:
            # Move batch to the appropriate device            
            outputs = model(**batch)        
            logits = outputs.logits
            labels = batch['labels']
            
            # Shift logits and labels to align them properly
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the logits and labels to calculate the loss
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)

            # Create a mask to ignore the padding tokens (-100) in loss calculation
            valid_mask = flat_labels != -100
            
            # Apply the mask to filter out invalid entries
            filtered_logits = flat_logits[valid_mask]
            filtered_labels = flat_labels[valid_mask]

            # Calculate the loss for valid entries
            loss = xent_loss(filtered_logits, filtered_labels)
            total_loss += loss

            # Calculate top-k accuracies
            top_k_values, top_k_indices = torch.topk(filtered_logits, k=max(top_k_accuracies.keys()), dim=-1)
            expanded_labels = filtered_labels.unsqueeze(1)
            
            correct_predictions = top_k_indices == expanded_labels
            for k in top_k_accuracies:
                top_k_accuracies[k] += correct_predictions[:, :k].sum()

            
            # Update the total count of evaluated tokens
            total_evaluated_tokens += valid_mask.sum()
    
    total_evaluated_tokens = torch.sum(accelerator.gather(total_evaluated_tokens).detach().cpu()).item()
    total_loss = torch.sum(accelerator.gather(total_loss).detach().cpu()).item()
    # Normalize the top-k accuracies by the total number of evaluated tokens
    for k in top_k_accuracies:
        correct_tokens = torch.sum(accelerator.gather(top_k_accuracies[k]).detach().cpu()).item()
        top_k_accuracies[k] = correct_tokens / total_evaluated_tokens if total_evaluated_tokens > 0 else 0

    # Calculate the average loss
    avg_loss = total_loss / total_evaluated_tokens if len(eval_loader) > 0 else 0.0

    return avg_loss, top_k_accuracies


def save_checkpoint(model, optimizer, accelerator, epoch, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': accelerator.unwrap_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accelerator_rng_state': accelerator.state
    }
    # Save the checkpoint
    accelerator.save(checkpoint, filename)
def add_module_prefix(state_dict):
    """Add 'module.' prefix to state dict keys if not present."""
    return {('module.' + k if not k.startswith('module.') else k): v for k, v in state_dict.items()}

def load_checkpoint(model, optimizer, accelerator, filename="checkpoint.pth"):
    checkpoint = torch.load(filename, map_location=accelerator.device)
    adjusted_model_state_dict = add_module_prefix(checkpoint['model_state_dict'])
    model.load_state_dict(adjusted_model_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

def find_latest_checkpoint(checkpoint_dir):
    # List all files in the directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint") and f.endswith(".pth")]
    # Sort files by epoch number in descending order
    checkpoint_files.sort(key=lambda x: int(x.replace('checkpoint', '').replace('.pth', '')), reverse=True)
    # Return the latest checkpoint file
    return checkpoint_files[0] if checkpoint_files else None

def main(args, output_dir, train_texts, eval_texts):
    # model setup
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2", cache_dir=args.cachedir)
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", cache_dir=args.cachedir)
    vocab_size = tokenizer.vocab_size
    cached_embedding = model.transformer.wte.weight[:vocab_size]
    dim = model.transformer.wte.weight.shape[1]
    pad_idx = vocab_size
    extended_embedding = nn.Embedding(vocab_size + 1, dim, padding_idx=pad_idx)
    extended_weight = torch.cat([cached_embedding, torch.zeros(1, dim)])
    del cached_embedding
    extended_embedding.load_state_dict({"weight": extended_weight})
    model.transformer.wte = extended_embedding
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    cutoff_len = 64
    grad_accum_steps = 64
    total_epochs = 20
    batch_size = 256
    log_extension = f'log_models_and_accuracies'
    log_dir = os.path.join(output_dir, log_extension)
    accelerator = Accelerator()
    trial = args.trial

    if accelerator.is_main_process:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    accelerator.wait_for_everyone()
    datasets.disable_progress_bars()


    # tokenizer
    def tokenize(example):
        # Tokenizing the sentence and adding BOS and EOS tokens.
        sent = example['text']
        sent = tokenizer.tokenize(sent)
        sent = [tokenizer.bos_token] + sent + [tokenizer.eos_token]
        
        # Encoding the tokens to get 'input_ids' and 'attention_mask'
        encoded_dict = tokenizer.encode_plus(
            sent,
            max_length=cutoff_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            is_split_into_words=True
        )
        
        # Flatten the 'input_ids' and convert to long for consistency
        input_ids = encoded_dict['input_ids'].flatten().long()
        
        # Constructing 'labels' based on 'input_ids': ignoring padding tokens by setting them to -100
        labels = [-100 if token == tokenizer.pad_token_id else token for token in input_ids.tolist()]
        
        # Building the final result dictionary
        result = {
            'input_ids': input_ids.tolist(),
            'attention_mask': encoded_dict['attention_mask'].flatten().long().tolist(),
            'labels': labels
        }
        return result

    # dataset setup
    train_dict = [{'text': x} for x in train_texts]
    train_dataset_hf = Dataset.from_list(train_dict)
    train_data_tokenized = train_dataset_hf.shuffle().map(tokenize, num_proc=5)
    train_data_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    test_data_dict = [{'text': x} for x in eval_texts]
    test_dataset_hf = Dataset.from_list(test_data_dict)
    test_data_tokenized = test_dataset_hf.map(tokenize, num_proc=5)
    test_data_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    train_dataloader = DataLoader(train_data_tokenized, batch_size=batch_size, num_workers=16)
    test_dataloader = DataLoader(test_data_tokenized, batch_size=8, drop_last=False, shuffle=False)

    # accelerator
    ##########################################################################################
    # Fill in the location of the finetuned DistilGPT2 checkpoint, if you have one.
    # We trained a DistilGPT2 model on PrE-Text/data/initialization.json to get a pretrained checkpoint.
    pretrained_ckpt = './c4_checkpoint.pth'
    checkpoint = torch.load(pretrained_ckpt, map_location=accelerator.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    ##########################################################################################
    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, AdamW(model.parameters(), lr=0.0002), train_dataloader, test_dataloader
    )

    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')

    avg_loss, top_k_accuracies = evaluate(model, eval_loader, accelerator, cross_entropy_loss)
    if accelerator.is_main_process:
        print(f'No finetuning, evaluation Loss: {avg_loss:.4f}', file=sys.stderr)
        for k, accuracy in top_k_accuracies.items():
            print(f'No finetuning, Top-{k} Accuracy: {accuracy:.4f}', file=sys.stderr)


    start_epoch = 0
    best_accuracy = 0.
    best_dict = None
    # Run training and evaluation
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0
        num_accumulated_steps = 0  # Track the number of accumulated steps
        curr_step_loss = 0
        num_actual_steps = 1
        for step, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / grad_accum_steps  # Normalize loss to account for accumulation
            accelerator.backward(loss)
            total_loss += loss.item()
            curr_step_loss += loss.item()
            num_accumulated_steps += 1

            # Perform the optimization step at the specified accumulation interval or at the last batch
            if (step + 1) % grad_accum_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                num_accumulated_steps = 0  # Reset the accumulation counter after an optimizer step
                if accelerator.is_main_process:
                    print(f"Epoch {epoch}, Step {num_actual_steps} loss: {curr_step_loss}", file=sys.stderr)
                curr_step_loss = 0
                num_actual_steps += 1

        # Calculate average loss over actual number of updates to adjust for any smaller final accumulation
        actual_updates = len(train_loader) // grad_accum_steps + (1 if len(train_loader) % grad_accum_steps != 0 else 0)
        avg_loss = total_loss / actual_updates
        if accelerator.is_main_process:
            print(f"Epoch {epoch} Avg training loss: {avg_loss}", file=sys.stderr)


        avg_loss, top_k_accuracies = evaluate(model, eval_loader, accelerator, cross_entropy_loss)
        if accelerator.is_main_process:
            print(f'Epoch {epoch} evaluation Loss: {avg_loss:.4f}', file=sys.stderr)
            for k, accuracy in top_k_accuracies.items():
                print(f'Epoch {epoch} Top-{k} Accuracy: {accuracy:.4f}', file=sys.stderr)
            top_k_accuracies['cross_entropy_loss'] = avg_loss
            stats_path = os.path.join(log_dir, f'epoch{epoch}_stats.json')
            print('Saving stats in ', stats_path, file=sys.stderr)
            with open(stats_path, 'w+') as f:
                json.dump(top_k_accuracies, f)
            
            if best_accuracy < top_k_accuracies[1]:
                best_accuracy = top_k_accuracies[1]
                best_dict = top_k_accuracies

            checkpoint_path = os.path.join(log_dir, f'checkpoint{epoch}.pth')
            save_checkpoint(model, optimizer, accelerator, epoch, filename=checkpoint_path)
        
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        stats_path = os.path.join(log_dir, f'best_stats.json')
        print('Saving stats in ', stats_path, file=sys.stderr)
        with open(stats_path, 'w+') as f:
            json.dump(best_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaluate on downstream DistilGPT next token prediction task.')
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
    args = parser.parse_args()
    output_dir = os.path.join(args.outputdir,
        "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}/".format(
            args.datadir,
            args.mask,
            args.lookahead,
            args.multiplier * 256,
            args.t_steps,
            args.H_multiplier,
            args.sensitivity,
            args.sigma,
            args.delta,
            args.trial
        )
    )
    with open(os.path.join(f'./data/{args.datadir}_eval.json'), 'r', encoding='utf8') as f:
        test_data_raw = json.load(f)["1"]
    with open(os.path.join(output_dir, f'llama7b_text_syn.json'), 'r', encoding='utf8') as f:
        s = json.load(f)
    all_data = []
    for text in s:
        split_samples = re.split("Orig", text)
        raw_sample = split_samples[0]
        raw_sample = raw_sample.strip()
        raw_sample = raw_sample.strip('\n')
        if len(raw_sample.split(' ')) > 3:
            all_data.append(raw_sample.replace('\n\n', ' ').replace('\n', ' '))
    main(args, output_dir, all_data, test_data_raw)