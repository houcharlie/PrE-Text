# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-70B-chat-AWQ", cache_dir='/dev/shm/LLM/again')
# model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-70B-chat-AWQ", cache_dir='/dev/shm/LLM/again')
from transformers import pipeline
from tqdm import tqdm
from vllm import LLM, SamplingParams
import json
import random
import time
import argparse
import os


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
    parser.add_argument('-machine_num', type=int, default=0)
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
    sd_list = []
    for i in range(11):
        with open(os.path.join(output_dir, f'surviving_text_it{i}.json') , 'r', encoding='utf8') as f:
            sd_list.extend(json.load(f))
    print('Number of seeds', len(sd_list))
    single_prompt = "List of 6 diverse original text samples:\nOriginal Text Sample 1\n{0}\nOriginal Text Sample 2\n{1}\nOriginal Text Sample 3\n{2}\nOriginal Text Sample 4\n"
    prompt_list = []
    print('Run 50000 samples')
    os.path.join(output_dir, 'llama7b_text_syn_{0}.json'.format(args.machine_num))
    for _ in range(50000):
        examples = random.sample(sd_list, 3)
        curr_prompt = single_prompt.format(examples[0].replace('\n', ' ').replace('\t', ' '), examples[1].replace('\n', ' ').replace('\t', ' '), examples[2].replace('\n', ' ').replace('\t', ' '))
        prompt_list.append(curr_prompt)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=85)
    llm = LLM(model="meta-llama/Llama-2-7b-hf", download_dir='/ocean/projects/cis230033p/houc/LLM/llama_7b_model', max_model_len=1000, seed=args.machine_num * args.machine_num)
    outputs = llm.generate(prompt_list, sampling_params)
    # Print the outputs.
    output_list = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        output_list.append(generated_text)
    with open(os.path.join(output_dir, 'llama7b_text_syn_{0}.json'.format(args.machine_num)), 'w+', encoding="utf8") as f:
        json.dump(output_list, f, ensure_ascii=False)
