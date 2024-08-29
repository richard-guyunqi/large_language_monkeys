python llmonk/generate/MATH.py \
offset=0 \
gpus='0' \
limit=128 \
stride=2 \
num_prompts=2 \
num_samples=512 \
batch_size=16 \
model=meta-llama/Meta-Llama-3-8B-Instruct \
save_dir=/home/richard/Downloads/large_language_monkeys/results/ \
dataset='math' \
--list vllm_args \
--disable-log-requests list-- 