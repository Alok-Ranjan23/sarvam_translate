# Pitch: Add Sarvam-Translate as a Translation Backend

## Problem

Our translation pipeline uses IndicTrans2 (200M distilled), a sentence-level NMT model. It handles clean, isolated sentences well but cannot serve three growing user needs:

- **No document/paragraph translation.** Text must be split into sentences before translation. Cross-sentence context (pronouns, tone, terminology) is lost, producing disjointed output for help articles, product descriptions, and legal content.
- **Breaks on real-world input.** Code-mixed text (Hinglish), markdown, HTML, and informal language fail the preprocessing pipeline (script unification, sentence segmentation) or produce poor results.
- **No Indian-to-Indian translation.** Hindi→Tamil requires two hops (Hindi→English→Tamil), doubling latency and degrading quality. Users working across Indian languages have no direct path.

We want to add a second translation backend that fills these gaps without replacing IndicTrans2.

---

## Solution

Add **Sarvam-Translate** (Gemma-3 4B, GGUF quantized) as an opt-in backend in `qvac-lib-infer-nmtcpp`, running locally on CPU via `llama.cpp`.

**What it solves:**

| Gap | How Sarvam addresses it |
|-----|------------------------|
| Document translation | Translates full paragraphs/documents in one pass, preserving coherence |
| Messy input | LLM-based -- handles code-mixed, markdown, informal text without preprocessing |
| X→Y translation | Single model covers all 22 Indian languages, all directions (En→X, X→En, X→Y) |
| Style control | Instruction prompting ("translate formally", "keep brand names in English") |

**Integration:**

- New `LLAMA` backend behind a compile flag (`USE_LLAMA`) -- zero impact on existing Bergamot/IndicTrans2 paths
- `llama.cpp` added via `vcpkg`, same build system already in use
- JS layer routes requests to Sarvam or IndicTrans2 based on `ModelTypes`
- Model ships as a single GGUF file (2.3 GB for IQ4_XS, up to 4.5 GB for Q8_0)

**POC completed:** Python prototype with side-by-side benchmarks validating quality, speed, and memory. C++ integration plan documented with all interface changes specified.

- [POC plan](01_sarvam_vs_indictrans_poc.plan.md)
- [C++ integration plan](02_add_sarvam_to_nmtcpp.plan.md)

---

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Memory** -- 3.3 GB peak vs ~1 GB for IndicTrans2 | Excludes devices with < 4 GB RAM | Opt-in backend; IndicTrans2 remains default for constrained devices |
| **Speed** -- ~6 tok/s vs ~22 tok/s | Slower for short sentences | Route sentence-level requests to IndicTrans2; Sarvam used only where document context matters |
| **GPL-3.0 license** | Copyleft requirement if distributed in proprietary product | Isolate behind compile flag; legal review before shipping; consider API-only deployment |
| **Hallucination** | LLM may add/omit content or produce non-deterministic output | Constrained sampling (low temperature), output length validation, post-translation checks |
| **Low-resource languages** | Aggressive quantization (IQ4_XS) may degrade quality for Bodo, Santali, Kashmiri | Default to Q4_K_M or higher for production; benchmark per language before rollout |
| **Cold start** | 3-10s model load time | Keep model loaded in memory; warm pool for serverless |

---

## Out of scope

- **Replacing IndicTrans2.** This adds a second backend, not a replacement. IndicTrans2 remains the default for sentence-level, high-throughput translation.
- **GPU inference.** This pitch targets CPU-only deployment. GPU acceleration is a future optimization.
- **Training or fine-tuning.** We use the model as-is from Sarvam AI. No custom training.
- **Mobile/browser deployment.** The 4B model is too large for mobile or WASM. Out of scope for this phase.

---

## Nice to haves

- **Streaming translation output.** `llama.cpp` supports token-by-token streaming -- could show translation progressively for long documents.
- **Automatic backend selection.** Route short sentences to IndicTrans2 and long paragraphs to Sarvam automatically based on input length, without user intervention.
- **Quality benchmarking per language pair.** Systematic BLEU/COMET evaluation across all 22 languages to identify where Sarvam outperforms or underperforms IndicTrans2.
- **Higher quantization options.** Ship Q5_K_M or Q8_0 for users with sufficient RAM who want best quality.
- **X→Y translation UI.** Expose Indian-to-Indian language pairs (Hindi→Tamil, Bengali→Gujarati) in the product UI, which is currently not possible with IndicTrans2.
