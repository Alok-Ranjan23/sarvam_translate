"""
Sarvam-Translate wrapper using llama-cpp-python for CPU inference.

The model is a Gemma-3 4B fine-tune that translates via chat-style prompts:
  system: "Translate the text below to {language}."
  user:   "<source text>"
"""

import os
import glob
import time
from typing import Optional

from llama_cpp import Llama


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

LANGUAGE_NAMES = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
    "sa": "Sanskrit",
    "ne": "Nepali",
    "sd": "Sindhi",
    "ks": "Kashmiri",
    "doi": "Dogri",
    "kok": "Konkani",
    "mai": "Maithili",
    "mni": "Manipuri",
    "brx": "Bodo",
    "sat": "Santali",
    "en": "English",
}


GGUF_PREFERENCE = ["IQ4_XS", "Q4_K_M", "Q5_K_M", "Q8_0"]


def _find_gguf(models_dir: str) -> str:
    """Locate the best .gguf file under the sarvam subdirectory (prefer higher quality)."""
    pattern = os.path.join(models_dir, "sarvam-translate-gguf", "*.gguf")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No .gguf file found in {pattern}. Run download_models.py first."
        )
    for tag in GGUF_PREFERENCE:
        for m in matches:
            if tag in m:
                return m
    return sorted(matches)[0]


class SarvamTranslator:
    """Thin wrapper around llama-cpp-python for Sarvam-Translate."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 8192,
        n_threads: int = 0,
    ):
        if model_path is None:
            model_path = _find_gguf(MODELS_DIR)

        # n_threads=0 lets llama.cpp auto-detect optimal thread count
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,
            verbose=False,
        )

    def translate(
        self,
        text: str,
        target_lang: str = "hi",
        temperature: float = 0.01,
        max_tokens: int = 1024,
    ) -> dict:
        """
        Translate *text* into *target_lang*.

        Returns a dict with keys:
            translation  - the translated string
            time_seconds - wall-clock time for generation
            tokens       - number of completion tokens generated
        """
        lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)

        messages = [
            {"role": "system", "content": f"Translate the text below to {lang_name}."},
            {"role": "user", "content": text},
        ]

        t0 = time.perf_counter()
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = time.perf_counter() - t0

        choice = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        completion_tokens = usage.get("completion_tokens", len(choice.split()))

        return {
            "translation": choice.strip(),
            "time_seconds": elapsed,
            "tokens": completion_tokens,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sarvam-Translate CLI")
    parser.add_argument("text", help="Text to translate")
    parser.add_argument(
        "--target-lang", default="hi", help="Target language code (default: hi)"
    )
    parser.add_argument(
        "--model-path", default=None, help="Path to .gguf file (auto-detected if omitted)"
    )
    args = parser.parse_args()

    translator = SarvamTranslator(model_path=args.model_path)
    result = translator.translate(args.text, target_lang=args.target_lang)
    print(f"Translation: {result['translation']}")
    print(f"Time: {result['time_seconds']:.2f}s  |  Tokens: {result['tokens']}")
