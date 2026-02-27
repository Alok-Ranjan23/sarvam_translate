#!/usr/bin/env python3
"""Diagnose why Sarvam-Translate produces English for hi→ta."""

import os
from translate_sarvam import SarvamTranslator, _find_gguf, MODELS_DIR

gguf_path = _find_gguf(MODELS_DIR)
print(f"GGUF: {gguf_path}\n")

from llama_cpp import Llama
llm = Llama(model_path=gguf_path, n_ctx=8192, n_gpu_layers=0, verbose=False)

# Check what chat template the GGUF has
meta = llm.metadata
for k, v in meta.items():
    if "chat" in k.lower() or "template" in k.lower():
        print(f"[META] {k} = {v[:300]}...\n")

test_hi = "उन्होंने कहा कि अब हमारे पास 4 महीने उम्र वाले चूहे हैं।"

# --- Test 1: current approach (system + user) ---
print("=" * 60)
print("TEST 1: system + user via create_chat_completion")
print("=" * 60)
msgs = [
    {"role": "system", "content": "Translate the text below to Tamil."},
    {"role": "user", "content": test_hi},
]
resp = llm.create_chat_completion(messages=msgs, temperature=0.01, max_tokens=256)
out1 = resp["choices"][0]["message"]["content"]
print(f"Output: {out1}\n")

# --- Test 2: no system message, instruction in user turn ---
print("=" * 60)
print("TEST 2: single user message with instruction")
print("=" * 60)
msgs2 = [
    {"role": "user", "content": f"Translate the text below to Tamil.\n\n{test_hi}"},
]
resp2 = llm.create_chat_completion(messages=msgs2, temperature=0.01, max_tokens=256)
out2 = resp2["choices"][0]["message"]["content"]
print(f"Output: {out2}\n")

# --- Test 3: raw Gemma-3 template with system turn ---
print("=" * 60)
print("TEST 3: raw Gemma-3 prompt (manual template)")
print("=" * 60)
raw_prompt = (
    "<start_of_turn>system\n"
    "Translate the text below to Tamil.<end_of_turn>\n"
    "<start_of_turn>user\n"
    f"{test_hi}<end_of_turn>\n"
    "<start_of_turn>model\n"
)
resp3 = llm(raw_prompt, temperature=0.01, max_tokens=256, stop=["<end_of_turn>"])
out3 = resp3["choices"][0]["text"]
print(f"Output: {out3}\n")

# --- Test 4: raw prompt, instruction in target language ---
print("=" * 60)
print("TEST 4: instruction in Hindi (target: Tamil)")
print("=" * 60)
raw_prompt4 = (
    "<start_of_turn>system\n"
    "नीचे दिए गए पाठ का तमिल में अनुवाद करें।<end_of_turn>\n"
    "<start_of_turn>user\n"
    f"{test_hi}<end_of_turn>\n"
    "<start_of_turn>model\n"
)
resp4 = llm(raw_prompt4, temperature=0.01, max_tokens=256, stop=["<end_of_turn>"])
out4 = resp4["choices"][0]["text"]
print(f"Output: {out4}\n")

# --- Test 5: few-shot style with an example ---
print("=" * 60)
print("TEST 5: with explicit language tag")
print("=" * 60)
raw_prompt5 = (
    "<start_of_turn>system\n"
    "You are a Hindi to Tamil translator. Translate the given Hindi text to Tamil script. Do not translate to English.<end_of_turn>\n"
    "<start_of_turn>user\n"
    f"{test_hi}<end_of_turn>\n"
    "<start_of_turn>model\n"
)
resp5 = llm(raw_prompt5, temperature=0.01, max_tokens=256, stop=["<end_of_turn>"])
out5 = resp5["choices"][0]["text"]
print(f"Output: {out5}\n")

# --- Test 6: try Q4_K_M (less aggressive quantization) ---
import glob
q4km_files = glob.glob(
    os.path.join(MODELS_DIR, "sarvam-translate-gguf", "*Q4_K_M*")
)
if q4km_files:
    print("=" * 60)
    print(f"TEST 6: Q4_K_M model ({os.path.basename(q4km_files[0])})")
    print("=" * 60)
    del llm
    llm2 = Llama(model_path=q4km_files[0], n_ctx=8192, n_gpu_layers=0, verbose=False)
    msgs6 = [
        {"role": "system", "content": "Translate the text below to Tamil."},
        {"role": "user", "content": test_hi},
    ]
    resp6 = llm2.create_chat_completion(messages=msgs6, temperature=0.01, max_tokens=256)
    out6 = resp6["choices"][0]["message"]["content"]
    print(f"Output: {out6}\n")
    del llm2

print("=" * 60)
print("DONE -- compare which test produces Tamil output")
print("=" * 60)
