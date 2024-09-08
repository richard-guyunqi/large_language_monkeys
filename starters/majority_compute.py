# Load in the evaluation results as a score dict

from pathlib import Path
import yaml
from tqdm import tqdm
from pathlib import Path
import json

def load_yaml(path: Path):
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.CLoader)

    return data
    
# TODO: Set parameters here 
num_prompts = [1, 2]
num_samples = 32      # Number of samples from each prompt
majority_step = 2      # The step of increase for the range of majority
ranges = list(range(1, num_samples, majority_step))  


for num_prompt in num_prompts:
    # Directory containing YAML files
    scores = {}

    for range_count in tqdm(ranges):
        checks_dir = Path(f'majorities/math_eval_majority_{num_prompt}_{range_count}')

        # Iterate through all YAML files in the directory
        correct_count = 0
        total_count = 0

        for yaml_file in checks_dir.glob('*.yaml'):
            total_count += 1
            data = load_yaml(yaml_file)
            checks = data['is_corrects']
            if True in checks:
                correct_count += 1
                continue

        score = correct_count / total_count
        scores[range_count] = score

    
    save_dir = f'scores/majority_score_{num_prompt}.json'
    with open(f'{save_dir}', 'w') as file:
        json.dump(scores, file, indent=4)