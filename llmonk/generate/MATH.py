import torch
from datasets import load_dataset
from tqdm import tqdm
import pydra
import multiprocessing
import random
import requests
import json
from functools import partial

from llmonk.generate.prompts import MATH_COT_PROMPT
from llmonk.utils import save_yaml, GenerateScriptConfig
from llmonk.generate.vllm_utils import vllm_manager
from openai import OpenAI

def sample_prompts(prompt, num_prompts):
    prompts = []

    if num_prompts == 1:
        prompts.append(prompt)
    else:
        ## Set the API key
        with open('/home/richard/Downloads/blender/BlenderAlchemy/config/openai_apikey.txt', 'r') as file:
            api_key = file.read().strip()
        
        ## Set the API key
        client = OpenAI(api_key=api_key)
        MODEL='gpt-4o'

        while len(prompts) < num_prompts:
            try:
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that understands a prompt and can re-write it."},
                        {"role": "user", "content": 
                        f'''Paraphrase and re-write the following paragraph by SWAPPING SYNONYM. DO NOT LOSE ANY INFORMATION of it. Paragraph starts here:   
                        {prompt}
                        
                        '''}
                    ]
                )
                response = completion.choices[0].message.content
                prompts.append(response)
            except Exception as e:
                print(f'Error: {e}')
                continue
    
    with open('/home/richard/Downloads/large_language_monkeys/temporary.json', 'w') as file:
        json.dump(prompts, file, indent=4)
    return prompts


def run_inference(item, config: GenerateScriptConfig):

    # Set output path
    outpath = config.save_dir / f"{item['id']}.yaml"
    print(f'outpath: {outpath}')
    if outpath.exists():
        return

    # Generate prompt
    num_prompts = config.num_prompts
    base_prompt = MATH_COT_PROMPT + f"\n\nProblem:\n{item['problem']}\n\nSolution:"
    prompts = sample_prompts(base_prompt, num_prompts)

    # DEBUG:

    num_prompts = len(prompts)

    url = f"http://localhost:{config.vllm_port}/generate"

    num_samples = config.num_samples    # Number of times that we inference for that single problem
    batch_size = config.batch_size      # Number of times inferenced in a batch
    print(f'num_samples: {num_samples}, num_prompts: {num_prompts}, batch_size: {batch_size}')
    num_samples = int(num_samples / num_prompts)
    assert num_samples % batch_size == 0

    samples = []

    for prompt in prompts:
        # print(f'prompt: {prompt}')
        for _ in tqdm(range(num_samples // batch_size), desc=f"Item {item['id']}"):
            # print(f'len(prompt): {len(prompt)}')
            body = {
                "prompt": prompt,
                "max_tokens": config.max_tokens,
                "n": batch_size,
                "temperature": config.temperature,
                # "top_p": config.top_p,
                "stop": config.stop_strings,
                "logprobs": 1,
            }

            response = requests.post(url, json=body)
            respj = response.json()
            samples.extend(respj["text"])

    out = {
        "prompts": prompts,
        "question": item["problem"],
        "samples": samples,
        "gt_answer": item["solution"],
    }

    save_yaml(outpath, out)


@pydra.main(GenerateScriptConfig)
def main(
    config: GenerateScriptConfig,
):
    print(f'config.limit: {config.limit}')
    print(f'config.save_dir: {config.save_dir}')

# Load the test and training dataset. We mainly use testset, while training set is used for sampling few-shot
# cases only 
    test_dataset = list(
        load_dataset(
            "hendrycks/competition_math", "main", split="test", trust_remote_code=True
        )
    )
    train_dataset = list(
        load_dataset(
            "hendrycks/competition_math", "main", split="train", trust_remote_code=True
        )
    )

    print(f"Number of test items: {len(test_dataset)}")
    print(f"Number of train items: {len(train_dataset)}")

    random.seed(config.seed)

# Attach few-shot examples for each question in test-dataset
    for i, data in enumerate(train_dataset):
        data["id"] = i

    for i, data in enumerate(test_dataset):
        few_shot_items = random.sample(train_dataset, config.num_few_shot)
        data["id"] = i
        data["few_shot_items"] = few_shot_items

    random.shuffle(test_dataset)
    shuffled_limit = test_dataset

    if config.limit is not None:
        limit = config.limit
    else:
        limit = len(shuffled_limit)

    if config.stride is not None:
        stride = config.stride
    else:
        stride = 1

    if config.offset is not None:
        offset = config.offset
    else:
        offset = 0

    # Select the final set for evaluation. It samples test_dataset starting from index config.offset, 
    # ends at config.limit, and samples every config.stride 
    shuffled_limit = shuffled_limit[offset:limit:stride]

    print(f"Total number of items to process: {len(shuffled_limit)}")

    test_dataset = shuffled_limit

    with vllm_manager(config) as vllm_port:
        config.vllm_port = vllm_port

        go_func = partial(run_inference, config=config)

        if config.num_workers not in [0, None]:
            with multiprocessing.Pool(config.num_workers) as pool:
                predictions = list(
                    tqdm(
                        pool.imap_unordered(go_func, test_dataset),
                        total=len(test_dataset),
                    )
                )
        else:
            predictions = []
            for item in tqdm(test_dataset):
                predictions.append(go_func(item))


if __name__ == "__main__":
    main()
