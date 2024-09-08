'''
This file samples prompts from the base prompt and then samples responses from synonimized prompts.

Notes:
    This file utilizes GPU compute. 
'''

import subprocess
import os
import argparse
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run MATH generation script with specific GPU device.')
parser.add_argument('--gpu_device', type=str, default='0', help='Index for the GPU device to be used.')
args = parser.parse_args()

# Set environment variables
env = os.environ.copy()

gpu_device = args.gpu_device   # Use the command line argument for GPU device


# TODO: Set parameters
save_dir = 'results/'   # Results saved to {save_dir}/math_samples_{num_prompt}
num_prompts = [1, 2, 8, 32]
total_gpus = 3
num_samples = 1024    # Number of samples for each n-prompt case
batch_size = 16     # Batch size for LLama-3-8b calling
num_workers = 5   # Number of threads. On cluster, 3 gpus usually allows 6/5/5 threads for 1024/2048/4096. On Suzzane, 2 gpus allows 10/8/8 for 1024/2048/4096.
max_tokens = 1024

# # Parameters for Prompt 1.0 (Original)
# save_dir = 'results/'   # Results saved to {save_dir}/math_samples_{num_prompt}
# num_prompts = [1, 2, 8, 32]
# total_gpus = 3
# num_samples = 1024    # Number of samples for each n-prompt case
# batch_size = 16     # Batch size for LLama-3-8b calling
# num_workers = 5   # Number of threads. On cluster, 3 gpus usually allows 6/5/5 threads for 1024/2048/4096. On Suzzane, 2 gpus allows 10/8/8 for 1024/2048/4096.
# max_tokens = 512

# # Parameters for Prompt 2.0/3.0 
# save_dir = 'results/'   # Results saved to {save_dir}/math_samples_{num_prompt}
# num_prompts = [1, 2, 8, 32, 64]
# total_gpus = 3
# num_samples = 2048    # Number of samples for each n-prompt case
# batch_size = 16     # Batch size for LLama-3-8b calling
# num_workers = 5   # Number of threads. On cluster, 3 gpus usually allows 6/5/5 threads for 1024/2048/4096. On Suzzane, 2 gpus allows 10/8/8 for 1024/2048/4096.
# max_tokens = 512

for num_prompt in num_prompts:
    # offset and stride can be manually set to run MATH.py on different gpus  
    # For example, for two gpus, we set stride = 2 for both, offset = 0 and device = '0' for cuda:0, offset = 1 and device = '1' for cuda:1. 
    command = f'''python llmonk/generate/MATH.py \
        offset={gpu_device} \
        gpus='{gpu_device}' \
        limit=128 \
        stride={total_gpus} \
        num_prompts={num_prompt} \
        num_samples={num_samples} \
        num_workers={num_workers} \
        batch_size={batch_size} \
        model=meta-llama/Meta-Llama-3-8B-Instruct \
        save_dir={save_dir} \
        dataset='math' \
        --list vllm_args \
        --disable-log-requests list-- 
    '''

    result = subprocess.run(command, shell=True, env=env)

    if result.returncode == 0:
        print(f"Script executed successfully for num_prompt {num_prompt}")
    else:
        print(f"Script failed with return code {result.returncode} for num_prompt {num_prompt}")
