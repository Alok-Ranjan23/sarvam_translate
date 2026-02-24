"""
IndicTrans2 wrapper using HuggingFace transformers + IndicTransToolkit.

The model is a classical encoder-decoder (seq2seq) NMT system with
script-unification preprocessing.  Two separate checkpoints are needed:
  - en->indic : ai4bharat/indictrans2-en-indic-dist-200M
  - indic->en : ai4bharat/indictrans2-indic-en-dist-200M
"""

import os
import time
from typing import List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    raise ImportError(
        "IndicTransToolkit is required. Install it with:\n"
        "  git clone https://github.com/VarunGumma/IndicTransToolkit.git\n"
        "  pip install -e ./IndicTransToolkit"
    )


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

EN_INDIC_REPO = "ai4bharat/indictrans2-en-indic-dist-200M"
INDIC_EN_REPO = "ai4bharat/indictrans2-indic-en-dist-200M"

# IndicTrans2 uses Flores-200 style language codes
LANG_CODE_MAP = {
    "en":  "eng_Latn",
    "hi":  "hin_Deva",
    "bn":  "ben_Beng",
    "ta":  "tam_Taml",
    "te":  "tel_Telu",
    "mr":  "mar_Deva",
    "gu":  "guj_Gujr",
    "kn":  "kan_Knda",
    "ml":  "mal_Mlym",
    "pa":  "pan_Guru",
    "or":  "ory_Orya",
    "as":  "asm_Beng",
    "ur":  "urd_Arab",
    "sa":  "san_Deva",
    "ne":  "npi_Deva",
    "sd":  "snd_Arab",
    "ks":  "kas_Arab",
    "doi": "doi_Deva",
    "kok": "kok_Deva",
    "mai": "mai_Deva",
    "mni": "mni_Mtei",
    "brx": "brx_Deva",
    "sat": "sat_Olck",
}


def _resolve_lang(code: str) -> str:
    """Accept either short codes ('hi') or Flores codes ('hin_Deva')."""
    if "_" in code:
        return code
    if code in LANG_CODE_MAP:
        return LANG_CODE_MAP[code]
    raise ValueError(
        f"Unknown language code '{code}'. Use one of: {list(LANG_CODE_MAP.keys())}"
    )


class IndicTransTranslator:
    """Thin wrapper around IndicTrans2 200M distilled models."""

    def __init__(self, models_dir: Optional[str] = None):
        self.device = "cpu"
        self.models_dir = models_dir or MODELS_DIR
        self.ip = IndicProcessor(inference=True)

        self._en_indic_model = None
        self._en_indic_tokenizer = None
        self._indic_en_model = None
        self._indic_en_tokenizer = None

    def _load_en_indic(self):
        if self._en_indic_model is not None:
            return
        ckpt = self._ckpt_path(EN_INDIC_REPO)
        print(f"[IndicTrans2] Loading en->indic model from {ckpt} ...")
        self._en_indic_tokenizer = AutoTokenizer.from_pretrained(
            ckpt, trust_remote_code=True
        )
        self._en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt, trust_remote_code=True, low_cpu_mem_usage=True
        ).to(self.device)
        self._en_indic_model.eval()

    def _load_indic_en(self):
        if self._indic_en_model is not None:
            return
        ckpt = self._ckpt_path(INDIC_EN_REPO)
        print(f"[IndicTrans2] Loading indic->en model from {ckpt} ...")
        self._indic_en_tokenizer = AutoTokenizer.from_pretrained(
            ckpt, trust_remote_code=True
        )
        self._indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt, trust_remote_code=True, low_cpu_mem_usage=True
        ).to(self.device)
        self._indic_en_model.eval()

    def _ckpt_path(self, repo_id: str) -> str:
        """Return a local path if models were pre-downloaded, else the HF repo id."""
        local = os.path.join(self.models_dir, repo_id.split("/")[-1])
        if os.path.isdir(local) and os.listdir(local):
            return local
        return repo_id

    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "hi",
        num_beams: int = 5,
        max_length: int = 256,
    ) -> dict:
        """
        Translate *text* from *source_lang* to *target_lang*.

        Returns a dict with keys:
            translation  - the translated string
            time_seconds - wall-clock time for generation
            tokens       - number of tokens generated
        """
        src = _resolve_lang(source_lang)
        tgt = _resolve_lang(target_lang)

        is_en_to_indic = src == "eng_Latn"
        if is_en_to_indic:
            self._load_en_indic()
            model = self._en_indic_model
            tokenizer = self._en_indic_tokenizer
        else:
            self._load_indic_en()
            model = self._indic_en_model
            tokenizer = self._indic_en_tokenizer

        return self._run(
            [text], src, tgt, model, tokenizer, num_beams, max_length
        )

    def _run(
        self,
        sentences: List[str],
        src_lang: str,
        tgt_lang: str,
        model,
        tokenizer,
        num_beams: int,
        max_length: int,
    ) -> dict:
        batch = self.ip.preprocess_batch(
            sentences, src_lang=src_lang, tgt_lang=tgt_lang
        )

        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=1,
            )
        elapsed = time.perf_counter() - t0

        decoded = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        translations = self.ip.postprocess_batch(decoded, lang=tgt_lang)

        total_tokens = sum(len(seq) for seq in generated_tokens)

        return {
            "translation": translations[0] if len(translations) == 1 else translations,
            "time_seconds": elapsed,
            "tokens": total_tokens,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IndicTrans2 CLI")
    parser.add_argument("text", help="Text to translate")
    parser.add_argument(
        "--source-lang", default="en", help="Source language code (default: en)"
    )
    parser.add_argument(
        "--target-lang", default="hi", help="Target language code (default: hi)"
    )
    args = parser.parse_args()

    translator = IndicTransTranslator()
    result = translator.translate(
        args.text, source_lang=args.source_lang, target_lang=args.target_lang
    )
    print(f"Translation: {result['translation']}")
    print(f"Time: {result['time_seconds']:.2f}s  |  Tokens: {result['tokens']}")
