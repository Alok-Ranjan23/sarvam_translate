#!/usr/bin/env python3
"""
benchmark.py -- Side-by-side comparison of Sarvam-Translate vs IndicTrans2.

Measures translation output, wall-clock time, peak RSS memory, and tokens/sec
for each model, then prints a formatted comparison table.

Usage:
    python benchmark.py "Be the change you wish to see in the world."
    python benchmark.py --target-lang ta "Hello, how are you?"
    python benchmark.py --source-lang hi --target-lang en "नमस्ते दुनिया"
"""

import argparse
import os
import sys
import time
import traceback

import psutil
from tabulate import tabulate


def _mem_mb() -> float:
    """Current process RSS in megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _fmt_mem(mb: float) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.0f} MB"


def run_sarvam(text: str, source_lang: str, target_lang: str) -> dict:
    """Load Sarvam-Translate and run a single translation."""
    from translate_sarvam import SarvamTranslator

    mem_before = _mem_mb()
    translator = SarvamTranslator()
    mem_after_load = _mem_mb()

    result = translator.translate(text, target_lang=target_lang)
    mem_after_infer = _mem_mb()

    result["mem_model_mb"] = mem_after_load - mem_before
    result["mem_peak_mb"] = mem_after_infer - mem_before
    result["tokens_per_sec"] = (
        result["tokens"] / result["time_seconds"] if result["time_seconds"] > 0 else 0
    )

    del translator
    return result


def run_indictrans(text: str, source_lang: str, target_lang: str) -> dict:
    """Load IndicTrans2 and run a single translation."""
    from translate_indictrans import IndicTransTranslator

    mem_before = _mem_mb()
    translator = IndicTransTranslator()
    result = translator.translate(
        text, source_lang=source_lang, target_lang=target_lang
    )
    mem_after = _mem_mb()

    result["mem_peak_mb"] = mem_after - mem_before
    result["tokens_per_sec"] = (
        result["tokens"] / result["time_seconds"] if result["time_seconds"] > 0 else 0
    )

    del translator
    return result


def print_comparison(
    text: str,
    source_lang: str,
    target_lang: str,
    sarvam_result: dict | None,
    indictrans_result: dict | None,
):
    """Pretty-print a side-by-side comparison table."""
    width = 72
    print("\n" + "=" * width)
    print(f"  Input ({source_lang}): {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"  Target Language: {target_lang}")
    print("=" * width)

    rows = []

    def _val(result, key, fmt="{}"):
        if result is None:
            return "SKIPPED"
        return fmt.format(result[key])

    rows.append([
        "Translation",
        _val(sarvam_result, "translation"),
        _val(indictrans_result, "translation"),
    ])
    rows.append([
        "Time (sec)",
        _val(sarvam_result, "time_seconds", "{:.2f}"),
        _val(indictrans_result, "time_seconds", "{:.2f}"),
    ])
    rows.append([
        "Tokens generated",
        _val(sarvam_result, "tokens"),
        _val(indictrans_result, "tokens"),
    ])
    rows.append([
        "Tokens/sec",
        _val(sarvam_result, "tokens_per_sec", "{:.1f}"),
        _val(indictrans_result, "tokens_per_sec", "{:.1f}"),
    ])
    rows.append([
        "Peak Memory",
        _val(sarvam_result, "mem_peak_mb") if sarvam_result is None
        else _fmt_mem(sarvam_result["mem_peak_mb"]),
        _val(indictrans_result, "mem_peak_mb") if indictrans_result is None
        else _fmt_mem(indictrans_result["mem_peak_mb"]),
    ])

    print(tabulate(
        rows,
        headers=["Metric", "Sarvam-Translate (GGUF)", "IndicTrans2 (200M)"],
        tablefmt="grid",
        maxcolwidths=[18, 30, 30],
    ))
    print("=" * width + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: Sarvam-Translate vs IndicTrans2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("text", help="Text to translate")
    parser.add_argument(
        "--source-lang", default="en",
        help="Source language code (default: en)",
    )
    parser.add_argument(
        "--target-lang", default="hi",
        help="Target language code (default: hi)",
    )
    parser.add_argument(
        "--sarvam-only", action="store_true",
        help="Only run the Sarvam model",
    )
    parser.add_argument(
        "--indictrans-only", action="store_true",
        help="Only run the IndicTrans2 model",
    )
    args = parser.parse_args()

    sarvam_result = None
    indictrans_result = None

    if not args.indictrans_only:
        print("\n>>> Running Sarvam-Translate ...")
        try:
            sarvam_result = run_sarvam(args.text, args.source_lang, args.target_lang)
        except Exception:
            print("[Sarvam] FAILED:")
            traceback.print_exc()

    if not args.sarvam_only:
        print("\n>>> Running IndicTrans2 ...")
        try:
            indictrans_result = run_indictrans(
                args.text, args.source_lang, args.target_lang
            )
        except Exception:
            print("[IndicTrans2] FAILED:")
            traceback.print_exc()

    if sarvam_result is None and indictrans_result is None:
        print("\nBoth models failed. Check that models are downloaded (run download_models.py).")
        sys.exit(1)

    print_comparison(
        args.text, args.source_lang, args.target_lang,
        sarvam_result, indictrans_result,
    )


if __name__ == "__main__":
    main()
