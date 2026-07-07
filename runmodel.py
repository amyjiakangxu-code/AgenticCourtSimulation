# run_model.py
# This model is MLX-quantized (4-bit, group_size 64) for Apple Silicon,
# so it must be loaded with mlx-lm rather than transformers.
from mlx_lm import load, generate

# Shadow dir: symlinks to the real weights + a cleaned tokenizer_config.json
# (the original stores extra_special_tokens as a list, which transformers<5 rejects).
path = "/private/tmp/claude-501/-Users-amyxu-Developer-AgenticCourtSimulation/9acaffac-52d9-47df-9d49-eee5ceef7488/scratchpad/mlx_model"

model, tokenizer = load(path)

messages = [{"role": "user", "content": "Hello, who are you?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

text = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=True)
