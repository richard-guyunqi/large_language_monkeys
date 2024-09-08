# Usage
All files you need to run are stored in starters/. Due to dependency issues, make sure you run them under the directory large_language_monkeys/ otherwise errors might occur.

## Generation
First, you could pick the prompt version following the TODO in llmonk/generate/prompts.py and llmonk/generate/MATH.py. All you need to do is to comment out the code for other versions.

Then, use the starting code shown below:
```
python starters/generate.py --gpu_device [gpu_index, starting from 0]
```
All parameters, such as num_samples, num_prompts, total_gpus are set in starters/generate.py.
Note that we don't use multi-gpu inference that's slower than dividing the dataset and running on separate gpus. 


## Majority Pipeline
### Run majority votes
First, we take majority among different ranges by setting parameters in starters/take_majorities.py and running 
```
python starters/take_majorities.py
```
The input is the result from generation, and the output is the majority vote result for different ranges.
### Evaluate majority results
We evaluate the majority results by setting parameters in starters/evaluate_majority.py and running 
```
python starters/evaluate_majority.py
```
The input is the majority results, and the output is an evaluation of each of them (correct/incorrect).

### Compute majority scores
We compute majority scores by counting how many majority vote results are correct by setting parameters in starters/majority_compute.py and running 
```
python starters/majority_compute.py
```

## Coverage pipeline
### Evaluate coverage
Coverage results are evaluated by examining all samples of a problem. You can modify the parameters in starter/evaluate_coverage.py to align with generation.
```
python starters/evaluate_coverage.py
```
The input is the samples, and the output is whether a problem contains a correct sample.

### Compute coverage scores
Coverage scores are computed by calculating the percentage of problems that contain a correct sample.
```
python starters/coverage_compute.py
```

## Visualization
The default coverage scores are saved to results/, and majority scores are saved to majorities/. If that is unchanged, you can follow the TODO in starters/visuzation.ipynb and run all the cells sequentially to get the visualized graphs.




# Large Language Monkeys

This repository provides the accompanying code for [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787).

Specifically, the code needed to:
1. Generate samples from various datasets and models.
2. Evaluate the correctness of the samples.

Four datasets are supported:
- GSM8K
- MATH
- CodeContests
- MiniF2F-MATH

We use [vLLM](https://docs.vllm.ai/en/latest/index.html) to do inference, so any models that they support will work with our generation scripts.

## Installation

We use two different conda environments for this project, as the lean-dojo version we use requires Python 3.9.19.

### Environment for MiniF2F-MATH

```
conda create -n llmonk-minif2f python=3.9.19
pip install -r requirements_minif2f.txt
```
To run evaluation on this dataset, we additionally need to install lean4. To do this, follow the installation instructions for your system according to [this website](https://leanprover-community.github.io/get_started.html).

When prompted with 
```
Current installation options:

  default toolchain: stable
  modify PATH variable: yes

1) Proceed with installation (default)
2) Customize installation
3) Cancel installation
```
Choose 2, and change the default toolchain to: `4.3.0-rc2`.

### Evironment for everything except MiniF2F-MATH

```
conda create -n llmonk python=3.11.8
pip install -r requirements.txt
```

## Repository Structure

The repo is organized as follows:

```
large-language-monkeys/
├── llmonk/
│   ├── evaluate/
│   │   ├── gsm8k.py
│   │   ├── math.py
│   │   ├── code_contests.py
│   │   └── minif2f.py
│   ├── generate/
│   │   ├── gsm8k.py
│   │   ├── math.py
│   │   ├── code_contests.py
│   │   └── minif2f.py
│   └── tests/
│   │   ├── math_datasets.py
│   │   ├── code_contests.py
│   │   └── minif2f.py
├── README.md
└── requirements.txt
```

- `llmonk/evaluate/`: contains the code to evaluate dataset samples
- `llmonk/generate/`: contains the code to generate samples from a model
- `llmonk/tests/`: contains code to check the correctness of our evaluation scripts

Within each folder, there is a file for each of the supported datasets (note that the scripts for MATH and GSM8K are combined under "math_datasets" for evaluation and testing).

## Generation Scripts

These scripts are used to generate samples from a model for a dataset.

### Usage

Each file has two mandatory arguments:
1. `model`: the huggingface model to use to generate the samples (same string you would pass to `.from_pretrained`)
2. `save_dir`: the directory to save the samples

For the remaining optional arguments (ex. temperature, number of samples, batch size, vllm arguments), please see the `GenerateScriptConfig` class in `llmonk/utils.py`.

### Output Format

The samples are saved as YAML files (one YAML file per problem). Every dataset's YAML file contains the following keys:
- `prompt`: the prompt for the problem
- `question`: the current question for the problem
- `samples`: a list of samples for each problem

For GSM8K and MATH, there is the additional key:
- `gt_answer`: the dataset's ground truth answer for the problem

For CodeContests, there is the additional key:
- `test_cases`: dictionary with the following keys:
    - `input`: list of strings corresponding to test case inputs
    - `output`: list of strings corresponding to test case outputs

For MiniF2F-MATH, there is the additional key:
- `theorem_name`: the name of the theorem to be proven

## Evaluation Scripts

These scripts evaluate the correctness of the samples generated by the generation scripts.

### Usage

Each file has two mandatory arguments:
1. `samples_dir`: the directory to the samples
2. `save_dir`: the directory to save the evaluation results

For the remaining optional arguments (ex. number of workers), please see the `EvaluateScriptConfig` class in `llmonk/utils.py`.

### Output Format

The evaluation results are saved as YAML files (one YAML file per problem), in the same format as the samples generated by the generation scripts with the additional key:
- `is_correct`: a list of booleans indicating whether each sample is correct, `is_correct[i]` is True if and only if `samples[i]` is correct

## Testing Scripts

The `llmonk/tests/` directory contains unit tests to evaluate the correctness of the evaluation scripts.

## Example Commands

See `commands.md` for examples of how to run generation, evaluation, and testing.

## Citation

If you use this code in your research, please cite our paper. You can use the following BibTeX entry:

```bibtex
@misc{brown2024largelanguagemonkeysscaling,
      title={Large Language Monkeys: Scaling Inference Compute with Repeated Sampling}, 
      author={Bradley Brown and Jordan Juravsky and Ryan Ehrlich and Ronald Clark and Quoc V. Le and Christopher Ré and Azalia Mirhoseini},
      year={2024},
      eprint={2407.21787},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.21787}, 
}
```
