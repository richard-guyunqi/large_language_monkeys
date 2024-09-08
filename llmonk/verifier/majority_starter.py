# Turn generated responses into majority vote results, with range of majority as variable

import subprocess
import os
import argparse
from tqdm import tqdm

# Set environment variables
env = os.environ.copy()

ranges = list(range(1, 512, 20))

for range_count in tqdm(ranges):
    command = f'''CUDA_VISIBLE_DEVICES=0 python /orion/u/yrichard/large_language_monkeys/llmonk/verifier/math_majority.py \
    samples_dir=/orion/u/yrichard/large_language_monkeys/results/math_samples_32_new \
    save_dir=/orion/u/yrichard/large_language_monkeys/results/math_majority_32_new_{range_count} \
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