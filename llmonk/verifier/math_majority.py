from pathlib import Path
from tqdm import tqdm
import multiprocessing
import pydra
from copy import deepcopy
import re
from lm_eval.tasks.minerva_math.utils import (
    last_boxed_only_string,
    normalize_final_answer,
    get_unnormalized_answer,
    remove_boxed,
    is_equiv,
)

from llmonk.utils import load_yaml, save_yaml, EvaluateScriptConfig, MajorityScriptConfig


ANS_RE_GSM8k = re.compile(r"#### (\-?[\$0-9\.\,]+)")
INVALID_ANS_GSM8k = "[invalid]"
GSM8K_IGNORE_REGEXES = [",", "\\$", "\\.$"]


def filter_ignores(st, regexes_to_ignore):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            st = re.sub(s, "", st)
    return st


def extract_answer_gsm8k(completion):
    match = ANS_RE_GSM8k.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = filter_ignores(
            match_str,
            GSM8K_IGNORE_REGEXES,
        )
        return match_str
    else:
        return INVALID_ANS_GSM8k


def is_correct_gsm8k(model_completion, gt_example):
    gt_answer = extract_answer_gsm8k(gt_example)
    assert gt_answer != INVALID_ANS_GSM8k
    return extract_answer_gsm8k(model_completion) == gt_answer


def is_correct_minerva(og_pred, gt):
    pred = normalize_final_answer(get_unnormalized_answer(og_pred))
    # print(f'pred: {pred}')
    gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    # print(f'gt: {gt}')
    return is_equiv(pred, gt)

def derive_pred(og_pred):
    pred = normalize_final_answer(get_unnormalized_answer(og_pred))
    return pred

class ScriptConfig(MajorityScriptConfig):
    dset: str = "gsm8k"


def is_correct(sample: str, gt_answer: str, dset: str):
    if dset == "gsm8k":
        return is_correct_gsm8k(sample, gt_answer)
    elif dset == "math":
        return is_correct_minerva(sample, gt_answer)
    else:
        raise ValueError(f"Dataset {dset} not supported")


# def process_sample(config: ScriptConfig):
#     if config.save_path.exists():
#         return

#     result = load_yaml(config.sample_path)
#     corrects = []
#     print(f'len(result["samples"]): {len(result["samples"])}')

#     for sample in result["samples"]:
#         correct = is_correct(sample, result["gt_answer"], config.dset)
#         # print(f'correct: {correct}')
#         corrects.append(correct)

#     result["is_corrects"] = corrects

#     save_yaml(config.save_path, result)
    
def process_sample(config: ScriptConfig):
    if config.save_path.exists():
        return

    result = load_yaml(config.sample_path)
    votes = {}
    # print(f'len(result["samples"][:config.majority_range]): {len(result["samples"][:config.majority_range])}')

    for sample in result["samples"][:config.majority_range]:
        extracted_answer = derive_pred(sample)
        if extracted_answer not in votes.keys():
            votes[extracted_answer] = 0     
        votes[extracted_answer] += 1
    
    votes = dict(sorted(votes.items(), key=lambda item: item[1], reverse=True))
    for key in votes.keys():    
        if key not in ['[invalidanswer]', '?', '(Insertyouranswerhere)', '[Insertyouranswerhere]', '??', '', '(Insertyourfinalanswerhere)', '[Insertyouranswerhere.]', '(Insertanswerhere)', '[Insertanswerhere]']:
            majority_answer = key
            break
    # print(f'majority_answer: {majority_answer}')
    result["majority_sample"] = majority_answer

    save_yaml(config.save_path, result)


def get_tasks(config):

    # The yaml filenames of the samples
    sample_paths = Path(config.samples_dir).glob("*.yaml")

    tasks = []

    # Append config for each sample yaml file
    for sample_path in tqdm(sample_paths, desc="Loading generations"):
        save_path = config.save_dir / sample_path.name

        task_config = deepcopy(config)
        task_config.sample_path = sample_path
        task_config.save_path = save_path

        tasks.append(task_config)

    return tasks


@pydra.main(base=ScriptConfig)
def main(config: ScriptConfig):

    tasks = get_tasks(config)
    tasks = sorted(
        tasks, key=lambda x: x.save_path
    )  # sort so the same offset references the same tasks across machines

    # tasks also counted by offset, limit, and stride
    tasks = tasks[config.offset : config.limit : config.stride]

    print(f"Taking majority on {len(tasks)} problems.")

    if config.num_workers not in [0, None]:
        with multiprocessing.Pool(processes=config.num_workers) as pool:
            _ = list(tqdm(pool.map(process_sample, tasks), total=len(tasks)))
    else:
        for task in tqdm(tasks):
            process_sample(task)


if __name__ == "__main__":
    main()
