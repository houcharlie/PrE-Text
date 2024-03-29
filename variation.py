import torch


from accelerate import Accelerator
from custom_datasets import MatrixDataset
# from leven import levenshtein
from torch.utils.data import DataLoader

from transformers import RobertaForMaskedLM, RobertaTokenizer
from transformers.generation.utils import top_k_top_p_filtering


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
        # batch_size x seq_len
        # curr_inputs_pad_sums = torch.sum(curr_inputs == 1, axis=1)
        # former_inputs_pad_sums = torch.sum(inputs['input_ids'] == 1, axis=1)
        # attention_mask_pad_sums = torch.sum(inputs['attention_mask'] == 0, axis=1)
        # all_ok = torch.sum(curr_inputs_pad_sums - attention_mask_pad_sums) == 0
        # if not all_ok:

        #     accelerator.print(inputs['input_ids'])
        #     accelerator.print(curr_inputs)
        #     accelerator.print(inputs['attention_mask'])
        #     accelerator.print(former_inputs_pad_sums)
        #     accelerator.print(curr_inputs_pad_sums)
        #     accelerator.print(attention_mask_pad_sums)
        #     import ipdb; ipdb.set_trace()

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


