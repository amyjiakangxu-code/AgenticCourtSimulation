# runnondistilled.py — interactive REPL for the base (non-distilled) Qwen2.5-1.5B-Instruct.
# Run with the main venv:  venv/bin/python runnondistilled.py
from threading import Thread

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "You are a helpful legal assistant specializing in Canadian case law and statutes."

# Load fully onto one device (MPS/GPU/CPU). Avoids device_map="auto", which was
# offloading weights to disk ("Some parameters are on the meta device...") and
# slowing generation — a 1.5B model fits comfortably in memory.
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Loading model on {device} (downloads from HuggingFace on first run)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device)
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
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    thread = Thread(
        target=model.generate,
        kwargs={**inputs, "max_new_tokens": 512, "streamer": streamer},
    )
    thread.start()

    print("bot> ", end="", flush=True)
    reply = ""
    for token in streamer:
        print(token, end="", flush=True)
        reply += token
    thread.join()
    print("\n")
    messages.append({"role": "assistant", "content": reply})
