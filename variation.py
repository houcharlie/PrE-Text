import torch
from accelerate import Accelerator
from custom_datasets import MatrixDataset
# from leven import levenshtein
from torch.utils.data import DataLoader
from torch import Tensor

from transformers import RobertaForMaskedLM, RobertaTokenizer
# from transformers.generation.utils import top_k_top_p_filtering

def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits
class Variation:
    def collate_fn_tokenizer(inputs, num_mask_percentage, tokenizer):
        """
        input_ids: list of seq_len length token tensors
        num_mask: number of items to mask out
        """
        input_ids = torch.cat([x["input_ids"] for x in inputs], dim=0)
        attention_mask = torch.cat([x["attention_mask"] for x in inputs], dim=0)
        num_valid_tokens = (
            torch.min(torch.sum(attention_mask, dim=1)) * num_mask_percentage
        )
        num_mask = int(num_valid_tokens)
        input_ids = input_ids.clone()
        labels = input_ids.clone()
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix = torch.full(labels.shape, 1.0)
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices_col = torch.multinomial(probability_matrix, num_samples=num_mask)
        masked_indices_row = torch.arange(labels.shape[0])[:, None].expand(
            size=masked_indices_col.shape
        )
        # mask out the inputs
        input_ids[masked_indices_row, masked_indices_col] = tokenizer.mask_token_id
        collate_output = {
            "inputs": {"input_ids": input_ids, "attention_mask": attention_mask},
            "masked_indices_col": masked_indices_col,
            "masked_indices_row": masked_indices_row,
            "num_masks": num_mask,
        }
        return collate_output

    def sample(inputs, masked_indices_col, masked_indices_row, num_masks, config):
        model = config["model"]
        accelerator = config['accelerator']
        curr_inputs = inputs["input_ids"].clone().to(model.device)
        masked_indices_col = masked_indices_col.to(model.device)
        masked_indices_row = masked_indices_row.to(model.device)
        for i in range(num_masks):
            outputs = model(
                input_ids=curr_inputs, attention_mask=inputs["attention_mask"]
            ).logits
            # select one random mask to fill back in for each sample
            mask_token_logits = (
                outputs[masked_indices_row[:, i], masked_indices_col[:, i], :].float()
                / config["temperature"]
            )
            filtered_logits = top_k_top_p_filtering(
                mask_token_logits, top_k=config["top_k"], top_p=config["top_p"]
            )
            probs = torch.nn.functional.softmax(filtered_logits, dim=1)
            replacement_tokens = torch.multinomial(probs, num_samples=1)
            curr_inputs[
                masked_indices_row[:, i], masked_indices_col[:, i]
            ] = replacement_tokens.squeeze()

        return {"input_ids": curr_inputs, "attention_mask": inputs["attention_mask"]}

    def produce_variation(parent_set, variation_deg, config):
        accelerator = config["accelerator"]
        tokenizer = config["tokenizer"]
        t_steps = config['t_steps']
        curr_ids = parent_set['input_ids']
        curr_masks = parent_set['attention_mask']
        for t_step in range(t_steps):

            parent_dataset = MatrixDataset({'input_ids': curr_ids, 'attention_mask': curr_masks})
            population_dataloader = DataLoader(
                parent_dataset,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                collate_fn=lambda x: Variation.collate_fn_tokenizer(
                    x, variation_deg, tokenizer
                ),
            )
            population_dataloader = accelerator.prepare(population_dataloader)
            offspring = []
            for _, batch in enumerate(population_dataloader):
                with torch.no_grad():
                    outputs = Variation.sample(**batch, config=config)
                    output_ids = outputs['input_ids']
                    offspring.append(accelerator.gather(output_ids).cpu())
            produced_ids = torch.cat(offspring)
            curr_ids = produced_ids
            curr_masks = (~(curr_ids == 1)).long()
        return {'input_ids':curr_ids, 'attention_mask':curr_masks}


