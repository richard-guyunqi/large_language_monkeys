# Loading eval results of coverage

from pathlib import Path
import yaml
from tqdm import tqdm
from pathlib import Path
import json

def load_yaml(path: Path):
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.CLoader)

    return data

num_prompts = [1, 2, 4, 32]

for num_prompt in num_prompts:
    # Directory containing YAML files
    checks_dir = Path(f'/orion/u/yrichard/large_language_monkeys/results/math_eval_{num_prompt}_new')


    # Iterate through all YAML files in the directory
    x_values = list(range(1, 512, 20))
    results = {}

    for x in tqdm(x_values):

        correct_count = 0
        total_count = 0

        for yaml_file in checks_dir.glob('*.yaml'):
            total_count += 1
            data = load_yaml(yaml_file)
            checks = data['is_corrects']
            if True in checks[:x]:
                correct_count += 1
                continue
        
        results[x] = correct_count/total_count

    with open(f'/orion/u/yrichard/large_language_monkeys/scores/results_{num_prompt}_new.json', 'w') as file:
        json.dump(results, file, indent=4)
