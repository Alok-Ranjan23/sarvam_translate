"""
Download both models for local CPU inference:
  - Sarvam-Translate GGUF (Q4_K_M) via huggingface_hub
  - IndicTrans2 200M distilled (en->indic & indic->en) via transformers
"""

import os
import argparse

from huggingface_hub import hf_hub_download


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

SARVAM_REPO = "mradermacher/sarvam-translate-GGUF"
SARVAM_FILENAME = "sarvam-translate.Q4_K_M.gguf"

INDICTRANS_EN_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"
INDICTRANS_INDIC_EN = "ai4bharat/indictrans2-indic-en-dist-200M"


def download_sarvam(models_dir: str, filename: str = SARVAM_FILENAME):
    """Download the Sarvam-Translate GGUF quantized model."""
    print(f"[Sarvam] Downloading {SARVAM_REPO} / {filename} ...")
    path = hf_hub_download(
        repo_id=SARVAM_REPO,
        filename=filename,
        local_dir=os.path.join(models_dir, "sarvam-translate-gguf"),
    )
    print(f"[Sarvam] Model saved to: {path}")
    return path


def download_indictrans(models_dir: str):
    """Download IndicTrans2 200M distilled models (both directions)."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    for repo_id in [INDICTRANS_EN_INDIC, INDICTRANS_INDIC_EN]:
        short_name = repo_id.split("/")[-1]
        dest = os.path.join(models_dir, short_name)
        if os.path.isdir(dest) and os.listdir(dest):
            print(f"[IndicTrans2] {short_name} already present at {dest}, skipping.")
            continue

        print(f"[IndicTrans2] Downloading {repo_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo_id, trust_remote_code=True)

        tokenizer.save_pretrained(dest)
        model.save_pretrained(dest)
        print(f"[IndicTrans2] Model saved to: {dest}")


def main():
    parser = argparse.ArgumentParser(description="Download translation models for the POC")
    parser.add_argument(
        "--models-dir",
        default=MODELS_DIR,
        help=f"Directory to store downloaded models (default: {MODELS_DIR})",
    )
    parser.add_argument(
        "--sarvam-only", action="store_true", help="Only download the Sarvam model"
    )
    parser.add_argument(
        "--indictrans-only", action="store_true", help="Only download the IndicTrans2 models"
    )
    parser.add_argument(
        "--sarvam-quant",
        default=SARVAM_FILENAME,
        help=f"GGUF filename to download (default: {SARVAM_FILENAME})",
    )
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    if not args.indictrans_only:
        download_sarvam(args.models_dir, args.sarvam_quant)

    if not args.sarvam_only:
        download_indictrans(args.models_dir)

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
