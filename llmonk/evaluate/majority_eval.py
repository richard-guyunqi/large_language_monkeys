# Turn majority vote results into evaluation results

import subprocess
import os
import argparse
from tqdm import tqdm

# Set environment variables
env = os.environ.copy()

ranges = list(range(1, 512, 20))

for range_count in tqdm(ranges):
    command = f'''python /orion/u/yrichard/large_language_monkeys/llmonk/evaluate/math_datasets_majority.py \
    samples_dir=/orion/u/yrichard/large_language_monkeys/results/math_majority_32_new_{range_count} \
    save_dir=/orion/u/yrichard/large_language_monkeys/results/math_eval_majority_32_new_{range_count} \
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