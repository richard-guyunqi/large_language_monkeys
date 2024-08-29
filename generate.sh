python llmonk/generate/MATH.py \
offset=1 \
gpus='1' \
limit=128 \
stride=2 \
num_prompts=4 \
num_samples=1024 \
batch_size=16 \
model=meta-llama/Meta-Llama-3-8B-Instruct \
save_dir=/home/richard/Downloads/large_language_monkeys/results/ \
dataset='math' \
--list vllm_args \
--disable-log-requests list-- 