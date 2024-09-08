'''
This file computes the success rate of the majority results of n-prompt cases.

Notes:
    This file does not utilize GPU compute. 
    The run time is usually about 15 minutes for each n-prompt case with 52(1024/20 or 512/10) majority results, so one hour if len(num_prompts) = 4.
'''


import subprocess
import os
import argparse
from tqdm import tqdm

# Set environment variables
env = os.environ.copy()

# TODO: Parameters
num_samples = 32      # Number of samples from each prompt
majority_step = 2      # The step of increase for the range of majority
ranges = list(range(1, num_samples, majority_step))     
num_prompts = [1, 2]   

for num_prompt in num_prompts:
    for range_count in tqdm(ranges):
        samples_dir = f'majorities/math_majority_{num_prompt}_{range_count}'
        save_dir = f'majorities/math_eval_majority_{num_prompt}_{range_count}'
        command = f'''python llmonk/evaluate/math_datasets_majority.py \
        samples_dir={samples_dir} \
        save_dir={save_dir} \
        offset=0 \
        stride=1 \
        limit=128 \
        dset=math
        '''

        result = subprocess.run(command, shell=True, env=env)

        if result.returncode == 0:
            print(f"Script executed successfully for range {range_count}")
        else:
            print(f"Script failed with return code {result.returncode} for range {range_count}")