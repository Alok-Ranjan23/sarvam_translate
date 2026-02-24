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

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 2. Install dependencies (CPU-only PyTorch)

```bash
pip install -r requirements.txt
```

This installs PyTorch **CPU-only** (~200 MB) instead of the full CUDA bundle (~2 GB+), since this POC targets CPU inference.

### 3. Install IndicTransToolkit (required for IndicTrans2)

```bash
git clone https://github.com/VarunGumma/IndicTransToolkit.git
pip install -e ./IndicTransToolkit
```

### 4. Download NLTK data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 5. Download models

```bash
# Download both models (Sarvam GGUF ~2.5 GB + IndicTrans2 ~400 MB)
python download_models.py

# Or download individually
python download_models.py --sarvam-only
python download_models.py --indictrans-only
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
# Sarvam-Translate standalone
python translate_sarvam.py "Be the change you wish to see." --target-lang hi

# IndicTrans2 standalone
python translate_indictrans.py "Be the change you wish to see." --target-lang hi
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
======================================================================
  Input (en): Be the change you wish to see in the world.
  Target Language: hi
======================================================================
+------------------+-------------------------------+---------------------------+
| Metric           | Sarvam-Translate (GGUF)       | IndicTrans2 (200M)        |
+==================+===============================+===========================+
| Translation      | दुनिया में वह बदलाव बनो जो   | दुनिया में वो बदलाव बनिए  |
|                  | तुम देखना चाहते हो।           | जो आप देखना चाहते हैं।    |
+------------------+-------------------------------+---------------------------+
| Time (sec)       | ~12.00                        | ~0.80                     |
+------------------+-------------------------------+---------------------------+
| Tokens generated | ~25                           | ~18                       |
+------------------+-------------------------------+---------------------------+
| Tokens/sec       | ~2.0                          | ~22.5                     |
+------------------+-------------------------------+---------------------------+
| Peak Memory      | ~2.50 GB                      | ~400 MB                   |
+------------------+-------------------------------+---------------------------+
======================================================================
```

*(Values are approximate and vary by hardware.)*

---

## Quantization options for Sarvam-Translate

The default download is `Q4_K_M` (~2.5 GB). You can pick a different quantization:

```bash
# Smaller, faster, lower quality
python download_models.py --sarvam-quant "sarvam-translate.IQ4_XS.gguf"

# Larger, slower, higher quality
python download_models.py --sarvam-quant "sarvam-translate.Q8_0.gguf"
```

Available from `mradermacher/sarvam-translate-GGUF`:

| Quant | Size | Quality |
|-------|------|---------|
| IQ4_XS | ~1.5 GB | Low |
| Q4_K_M | ~2.5 GB | Medium (default) |
| Q5_K_M | ~3.0 GB | Medium-High |
| Q6_K | ~3.5 GB | High |
| Q8_0 | ~4.5 GB | Near-original |

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
    │   └── sarvam-translate.Q4_K_M.gguf
    ├── indictrans2-en-indic-dist-200M/
    └── indictrans2-indic-en-dist-200M/
```
