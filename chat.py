# chat.py — interactive REPL for the distilled Canadian-law Qwen model.
# Run with the MLX venv:  venv_mlx/bin/python chat.py
from mlx_lm import load, stream_generate

# Shadow dir: symlinks to the real weights + a cleaned tokenizer_config.json
# (the original stores extra_special_tokens as a list, which transformers<5 rejects).
PATH = "/private/tmp/claude-501/-Users-amyxu-Developer-AgenticCourtSimulation/9acaffac-52d9-47df-9d49-eee5ceef7488/scratchpad/mlx_model"

SYSTEM = "You are a helpful legal assistant specializing in Canadian case law and statutes."

print("Loading model...")
model, tokenizer = load(PATH)
print("Ready. Type your message. Commands: /reset  /exit\n")

messages = [{"role": "system", "content": SYSTEM}]

while True:
    try:
        user = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not user:
        continue
    if user in ("/exit", "/quit"):
        break
    if user == "/reset":
        messages = [{"role": "system", "content": SYSTEM}]
        print("(conversation reset)\n")
        continue

    messages.append({"role": "user", "content": user})
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    print("bot> ", end="", flush=True)
    reply = ""
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=512):
        print(resp.text, end="", flush=True)
        reply += resp.text
    print("\n")
    messages.append({"role": "assistant", "content": reply})
