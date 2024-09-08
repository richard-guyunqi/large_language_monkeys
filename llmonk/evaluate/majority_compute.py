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
    
# Directory containing YAML files
scores = {}

for range_count in tqdm(range(1, 512, 20)):
    checks_dir = Path(f'/orion/u/yrichard/large_language_monkeys/results/math_eval_majority_32_new_{range_count}')

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

with open('/orion/u/yrichard/large_language_monkeys/scores/scores_32_new.json', 'w') as file:
    json.dump(scores, file, indent=4)