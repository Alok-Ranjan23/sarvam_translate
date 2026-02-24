# Sarvam-Translate vs IndicTrans2 -- Local CPU POC

A command-line proof-of-concept that runs **Sarvam-Translate** (4B LLM, GGUF quantized) and **IndicTrans2** (200M distilled seq2seq) side-by-side on CPU, comparing translation quality, speed, and memory usage across Indian languages.

---

## Architecture Overview

| | Sarvam-Translate | IndicTrans2 |
|---|---|---|
| **Type** | Decoder-only causal LM (Gemma-3 4B fine-tune) | Encoder-Decoder seq2seq NMT |
| **Parameters** | ~4 B | 200 M (distilled) |
| **Inference** | llama-cpp-python (GGUF Q4_K_M) | HuggingFace transformers |
| **Context** | 8192 tokens | 256 tokens |
| **Languages** | 22 Indian + English | 22 Indian + English |
| **License** | GPL-3.0 | MIT |

### How each model translates

**Sarvam-Translate** works like a chatbot -- you give it a system prompt ("Translate to Hindi") and a user message (source text), and it autoregressively generates the translation token-by-token. It can handle long documents, markdown, and code-mixed text.

**IndicTrans2** is a purpose-built NMT system. Source text goes through script unification (mapping 22 scripts into 5 canonical ones), gets encoded by a transformer encoder, then decoded into the target language. It is smaller, faster, and generally more accurate for sentence-level translation.

---

## Setup

### 1. Create a virtual environment and install dependencies

```bash
python -m venv ~/sarvam_venv
source ~/sarvam_venv/bin/activate
pip install -r requirements.txt
```

This creates the venv in your home directory for faster I/O and installs PyTorch **CPU-only** (~200 MB) instead of the full CUDA bundle (~2 GB+), since this POC targets CPU inference.

### 2. Install IndicTransToolkit (required for IndicTrans2)

```bash
git clone https://github.com/VarunGumma/IndicTransToolkit.git
pip install -e ./IndicTransToolkit
```

### 3. Download NLTK data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## Downloading Models

### Sarvam-Translate (GGUF)

Sarvam-Translate is an **open model** -- no login or approval needed. It is downloaded as a single GGUF file from [mradermacher/sarvam-translate-GGUF](https://huggingface.co/mradermacher/sarvam-translate-GGUF) on HuggingFace.

**Download the default quantization (Q4_K_M, ~2.5 GB):**

```bash
python download_models.py --sarvam-only
```

**Or pick a different quantization:**

```bash
# Smallest & fastest (~2.3 GB)
python download_models.py --sarvam-only --sarvam-quant "sarvam-translate.IQ4_XS.gguf"

# Higher quality (~3.0 GB)
python download_models.py --sarvam-only --sarvam-quant "sarvam-translate.Q5_K_M.gguf"

# Near-original quality (~4.5 GB)
python download_models.py --sarvam-only --sarvam-quant "sarvam-translate.Q8_0.gguf"
```

**Available quantizations:**

| Quant | Size | Quality | Best for |
|-------|------|---------|----------|
| IQ4_XS | ~2.3 GB | Good | Quick testing, low RAM |
| Q4_K_M | ~2.5 GB | Better (default) | Recommended balance |
| Q5_K_M | ~3.0 GB | High | Better translation accuracy |
| Q6_K | ~3.5 GB | Very High | Quality-focused |
| Q8_0 | ~4.5 GB | Near-original | Best quality, more RAM |

You can download multiple quantizations and switch between them using `--model-path`:

```bash
python translate_sarvam.py "Hello world" --target-lang hi --model-path models/sarvam-translate-gguf/sarvam-translate.Q4_K_M.gguf
```

### IndicTrans2 (200M Distilled)

IndicTrans2 models are **gated** on HuggingFace. You must request access and log in before downloading.

**Step 1: Request access (one-time)**

1. Visit [indictrans2-en-indic-dist-200M](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M) and click **"Agree and access repository"**
2. Visit [indictrans2-indic-en-dist-200M](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M) and do the same
3. Wait for approval (can be instant or take a few hours)

**Step 2: Log in to HuggingFace**

```bash
python -c "from huggingface_hub import login; login()"
```

Paste your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (create a **Read** token if you don't have one).

**Step 3: Download the models (~400 MB total for both directions)**

```bash
python download_models.py --indictrans-only
```

This downloads two checkpoints:
- `indictrans2-en-indic-dist-200M` — English to any Indian language
- `indictrans2-indic-en-dist-200M` — Any Indian language to English

### Download both models at once

```bash
python download_models.py
```

---

## Usage

### Side-by-side benchmark (recommended)

```bash
# English to Hindi (default)
python benchmark.py "Be the change you wish to see in the world."

# English to Tamil
python benchmark.py --target-lang ta "Hello, how are you?"

# Hindi to English
python benchmark.py --source-lang hi --target-lang en "नमस्ते दुनिया"

# Run only one model
python benchmark.py --sarvam-only "Hello world"
python benchmark.py --indictrans-only "Hello world"
```

### Individual model wrappers

```bash
# Sarvam-Translate standalone (en->hi)
python translate_sarvam.py "Be the change you wish to see." --target-lang hi

# Sarvam-Translate (hi->en) -- no --source-lang needed, the LLM auto-detects
python translate_sarvam.py "नमस्ते दुनिया" --target-lang en

# IndicTrans2 standalone (en->hi)
python translate_indictrans.py "Be the change you wish to see." --target-lang hi

# IndicTrans2 (hi->en) -- requires --source-lang since it uses a separate model per direction
python translate_indictrans.py "नमस्ते दुनिया" --source-lang hi --target-lang en
```

### Supported language codes

| Code | Language | Code | Language |
|------|----------|------|----------|
| `en` | English | `ml` | Malayalam |
| `hi` | Hindi | `pa` | Punjabi |
| `bn` | Bengali | `or` | Odia |
| `ta` | Tamil | `as` | Assamese |
| `te` | Telugu | `ur` | Urdu |
| `mr` | Marathi | `sa` | Sanskrit |
| `gu` | Gujarati | `ne` | Nepali |
| `kn` | Kannada | `sd` | Sindhi |

Additional codes: `ks` (Kashmiri), `doi` (Dogri), `kok` (Konkani), `mai` (Maithili), `mni` (Manipuri), `brx` (Bodo), `sat` (Santali).

---

## Expected output

```
========================================================================
  Input (en): Be the change you wish to see.
  Target Language: hi
========================================================================
+------------------+------------------------------+---------------------------+
| Metric           | Sarvam-Translate (GGUF)      | IndicTrans2 (200M)        |
+==================+==============================+===========================+
| Translation      | आप जो बदलाव देखना चाहते हैं, | वह बदलाव बनें जो आप देखना |
|                  | वह बदलाव खुद बनें।           | चाहते हैं।                |
+------------------+------------------------------+---------------------------+
| Time (sec)       | 2.10                         | 1.45                      |
+------------------+------------------------------+---------------------------+
| Tokens generated | 12                           | 11                        |
+------------------+------------------------------+---------------------------+
| Tokens/sec       | 5.7                          | 7.6                       |
+------------------+------------------------------+---------------------------+
| Peak Memory      | 3.24 GB                      | 986 MB                    |
+------------------+------------------------------+---------------------------+
========================================================================
```

*(Values vary by hardware, quantization choice, and CPU thread count.)*

---

## Performance tuning

Both models use OpenMP for multi-threaded inference. Pin threads to cores for best throughput:

```bash
OMP_NUM_THREADS=16 GOMP_CPU_AFFINITY=0-15 python benchmark.py "Be the change you wish to see."
```

| Variable | What it does |
|----------|-------------|
| `OMP_NUM_THREADS=N` | Limit OpenMP threads to N (set to your physical core count) |
| `GOMP_CPU_AFFINITY=0-N` | Pin threads to specific cores, prevents OS thread migration and cache thrashing |

**Determine your core count:**

```bash
nproc
```

**Auto-detect and run:**

```bash
OMP_NUM_THREADS=$(nproc) GOMP_CPU_AFFINITY="0-$(($(nproc)-1))" python benchmark.py "Be the change you wish to see."
```

> **Note on NUMA:** If your system uses NPS1 (single NUMA node), `numactl` is unnecessary -- all cores and memory are already unified. For NPS2/NPS4 configurations, bind to a single node:
> ```bash
> numactl --membind=0 --cpunodebind=0 python benchmark.py "Be the change you wish to see."
> ```

---

## File structure

```
sarvam_translate/
├── requirements.txt          # Python dependencies
├── download_models.py        # Model downloader script
├── translate_sarvam.py       # Sarvam-Translate wrapper (llama-cpp-python)
├── translate_indictrans.py   # IndicTrans2 wrapper (transformers)
├── benchmark.py              # Side-by-side comparison CLI
├── README.md                 # This file
├── IndicTransToolkit/        # Cloned toolkit (after setup)
└── models/                   # Downloaded models (after setup)
    ├── sarvam-translate-gguf/
    │   ├── sarvam-translate.IQ4_XS.gguf
    │   └── sarvam-translate.Q4_K_M.gguf
    ├── indictrans2-en-indic-dist-200M/
    └── indictrans2-indic-en-dist-200M/
```

---

## Why add Sarvam-Translate for Indian language translation

1. **Single model, all directions.** One GGUF file handles En→X and X→En for all 22 scheduled Indian languages. IndicTrans2 requires separate checkpoints per direction (en-indic vs indic-en), doubling storage and loading complexity.

2. **Document-level coherence.** Sarvam-Translate processes entire paragraphs or documents in one pass, maintaining context, coreferences, and tone across sentences. IndicTrans2 operates sentence-by-sentence, which can lose cross-sentence context.

3. **Handles messy, real-world input.** As a fine-tuned LLM (Gemma-3 4B), it gracefully handles code-mixed text (Hinglish), markdown, bullet lists, and informal language -- inputs that trip up traditional NMT pipelines requiring clean, segmented sentences.

4. **No preprocessing pipeline.** IndicTrans2 needs script unification, tokenization via IndicProcessor, and language-specific pre/post-processing. Sarvam-Translate takes raw text and a target language -- no external toolkit required.

5. **Bidirectional without explicit source language.** The LLM auto-detects the source language, so users don't need to specify `--source-lang`. This simplifies integration and handles multilingual input naturally.

6. **Quantization flexibility.** GGUF format offers a range of quantizations (IQ4_XS at 2.3 GB to Q8_0 at 4.5 GB), letting you trade quality vs. RAM on constrained hardware. The model runs well on CPU without GPU.

7. **Richer, more natural translations.** The generative approach produces translations with greater vocabulary diversity (higher type-token ratio) and more natural phrasing, especially for longer or stylistically complex text.

8. **Built for Indian languages from the ground up.** Sarvam AI is an Indian AI company focused specifically on Indian language technology. The model is trained and fine-tuned with Indian language data, cultural context, and usage patterns -- not a general-purpose multilingual model adapted after the fact.

9. **Active development and ecosystem.** Sarvam AI offers a broader product suite (speech-to-text, text-to-speech, transliteration, LLMs) that can be composed into end-to-end Indian language pipelines, with ongoing model updates and community support.

10. **Complementary to existing NMT.** Sarvam-Translate doesn't need to replace IndicTrans2 -- it can serve as a second backend. Use IndicTrans2 for high-throughput, low-latency sentence translation, and Sarvam for document-level, context-aware translation where quality matters more than speed.

---

## Challenges with Sarvam-Translate GGUF on different devices

1. **High memory footprint.** Even the smallest quantization (IQ4_XS) loads ~2.3 GB into RAM, with peak usage reaching ~3.3 GB during inference (model + KV cache). This rules out low-RAM devices like Raspberry Pi, budget phones, and embedded systems. IndicTrans2 (200M) peaks at under 1 GB.

2. **Slow inference on weak CPUs.** The POC showed ~6 tokens/sec on a capable server CPU. On a laptop i5, a phone ARM chip, or a low-power edge device, expect 1-3 tokens/sec or worse -- translating a paragraph could take 30+ seconds, making it impractical for real-time use.

3. **Architecture-specific compilation.** `llama.cpp` (the GGUF runtime) must be compiled per target architecture. x86_64 with AVX2, x86 without AVX, ARM64 with NEON, Apple Silicon -- each needs a separate build. SIMD mismatch (e.g., running an AVX2 build on an older CPU without AVX2) causes crashes or silent fallback to scalar code (much slower).

4. **No efficient batching.** Sarvam is a causal LLM -- it generates one translation at a time, sequentially. You can't batch 100 sentences and get parallel translations like IndicTrans2. For high-throughput pipelines (e.g., translating thousands of UI strings), this is a major bottleneck.

5. **Non-deterministic / hallucination risk.** Being generative, Sarvam can add explanatory text instead of just translating, hallucinate content not in the source, repeat or truncate output, or produce different translations for the same input across runs (temperature-dependent). This is a production reliability concern, especially for automated pipelines with no human review.

6. **Quantization quality varies by language.** IQ4_XS works reasonably for Hindi (high-resource), but for low-resource languages like Bodo, Santali, Dogri, or Kashmiri, the aggressive quantization may cause noticeable quality degradation. Higher quants (Q8_0) need more RAM, defeating the purpose on constrained devices.

7. **Context window vs memory tradeoff.** The model supports 131K context but the POC uses 8192 tokens. Increasing context for longer documents linearly increases KV cache memory. On a 4 GB device, you may be limited to very short context windows, negating the "document-level" advantage.

8. **GPL-3.0 license.** Sarvam-Translate is GPL-3.0 licensed. If you integrate it into a proprietary product, the copyleft clause requires you to open-source the entire work. This is a legal blocker for many commercial deployments. IndicTrans2 is MIT licensed -- no such restriction.

9. **Mobile / browser deployment is impractical.** On mobile, 2.3 GB model download + 3.3 GB peak RAM makes it unusable on most phones. In browser/WASM, `llama.cpp` has WASM support but 4B model inference is extremely slow and memory-constrained. On IoT/edge, most devices have 512 MB - 2 GB RAM total.

10. **Cold start time.** Loading a 2.3-4.5 GB GGUF model from disk takes 3-10 seconds depending on storage speed (much worse on cross-filesystem paths or HDD). IndicTrans2 loads in ~1-2 seconds. For serverless or on-demand translation, this cold start is painful.

**Summary:**

| Problem | Impact | Mitigation |
|---------|--------|------------|
| High RAM (~3.3 GB peak) | Excludes low-RAM devices | Use IndicTrans2 on constrained devices |
| Slow on weak CPUs | Poor UX for real-time | Offload to server, or use NMT locally |
| Per-architecture compilation | Build/deploy complexity | CI matrix for each target arch |
| No batching | Low throughput | Queue + sequential, or use NMT for bulk |
| Hallucination risk | Unreliable in automation | Post-validation, constrained sampling |
| Low-resource language quality | Worse with aggressive quant | Use Q5_K_M+ for low-resource langs |
| GPL-3.0 license | Blocks proprietary use | Legal review, or use API instead |
| Mobile/browser unusable | No edge deployment | Server-side only, or API |
| Cold start (3-10s) | Bad for serverless | Keep model loaded, warm pool |

The bottom line: Sarvam-Translate excels on servers or workstations with 8+ GB RAM and decent CPUs, but faces real challenges on mobile, edge, browser, and low-resource device targets. A **dual-backend approach** (Sarvam for quality, IndicTrans2 for speed/portability) is the practical answer.
