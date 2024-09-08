'''
This file computes the coverage of the samples of n-prompt cases.

Notes:
    This file does not utilize GPU compute. 
    The run time is usually below 5 minutes for each n-prompt case, so below 20 min if len(num_prompts) = 4.
'''


import subprocess
import os
import argparse
from tqdm import tqdm

# Set environment variables
env = os.environ.copy()

# TODO: Parameters
num_prompts = [1, 2, 8, 32, 64]     


for num_prompt in num_prompts:
    samples_dir = f'results/math_samples_{num_prompt}'      # Input dir (samples from generation)
    save_dir = f'results/math_eval_coverage_{num_prompt}'     # Output dir (majorities over samples) 

    command = f'''python llmonk/evaluate/math_datasets.py \
    samples_dir={samples_dir} \
    save_dir={save_dir} \
    offset=0 \
    stride=1 \
    limit=128 \
    dset=math
    '''

    result = subprocess.run(command, shell=True, env=env)

    if result.returncode == 0:
        print(f"Script executed successfully for num_prompt {num_prompt}")
    else:
        print(f"Script failed with return code {result.returncode} for num_prompt {num_prompt}")