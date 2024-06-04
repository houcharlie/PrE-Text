from transformers import pipeline
from tqdm import tqdm
from vllm import LLM, SamplingParams
import json
import random
import time
import argparse
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Run expansion.')
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
    sd_list = []
    for i in range(11):
        with open(os.path.join(output_dir, f'surviving_text_it{i}.json') , 'r', encoding='utf8') as f:
            sd_list.extend(json.load(f))
    llm = LLM(model="meta-llama/Llama-2-7b-hf", download_dir=args.cachedir, max_model_len=1000)
    print(output_dir, file=sys.stderr)
    print('Number of seeds', len(sd_list))
    single_prompt = "List of 6 diverse original text samples:\nOriginal Text Sample 1\n{0}\nOriginal Text Sample 2\n{1}\nOriginal Text Sample 3\n{2}\nOriginal Text Sample 4\n"
    prompt_list = []
    os.path.join(output_dir, 'llama7b_text_syn.json')
    for _ in range(50000):
        examples = random.sample(sd_list, 3)
        curr_prompt = single_prompt.format(examples[0].replace('\n', ' ').replace('\t', ' '), examples[1].replace('\n', ' ').replace('\t', ' '), examples[2].replace('\n', ' ').replace('\t', ' '))
        prompt_list.append(curr_prompt)
    print(prompt_list[0])
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=85)
    outputs = llm.generate(prompt_list, sampling_params)
    # Print the outputs.
    output_list = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        output_list.append(generated_text)
    with open(os.path.join(output_dir, 'llama7b_text_syn.json'), 'w+', encoding="utf8") as f:
        json.dump(output_list, f, ensure_ascii=False)
