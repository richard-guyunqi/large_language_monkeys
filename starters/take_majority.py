'''
This file turns generated samples into majority vote results, with range of majority as variable. 

Notes:
    This file does not utilize GPU compute. 
    The run time is usually about 15 minutes for an n-prompt case with 1024 samples, so 1 hour if len(num_prompts) = 4.
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
        samples_dir = f'results/math_samples_{num_prompt}'
        save_dir = f'majorities/math_majority_{num_prompt}_{range_count}'

        command = f'''python llmonk/verifier/math_majority.py \
        samples_dir={samples_dir} \
        save_dir={save_dir} \
        offset=0 \
        stride=1 \
        limit=128 \
        majority_range={range_count} \
        dset=math
        '''

        result = subprocess.run(command, shell=True, env=env)

        if result.returncode == 0:
            print(f"Script executed successfully for range {range_count}")
        else:
            print(f"Script failed with return code {result.returncode} for range {range_count}")