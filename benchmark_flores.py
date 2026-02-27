#!/usr/bin/env python3
"""
benchmark_flores.py -- Evaluate Sarvam-Translate and IndicTrans2 on the
FLORES-200 devtest dataset with reference-based BLEU, chrF++, and COMET.

Usage:
    python benchmark_flores.py                          # all 6 pairs
    python benchmark_flores.py --pairs en-hi hi-en      # specific pairs only
    python benchmark_flores.py --limit 100              # first 100 sentences
    python benchmark_flores.py --skip-comet             # skip COMET (faster)
    python benchmark_flores.py --sarvam-only            # only Sarvam
"""

import argparse
import json
import os
import sys
import time

from tabulate import tabulate

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "flores_results")

FLORES_LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
}

ALL_PAIRS = ["en-hi", "hi-en", "en-ta", "ta-en", "hi-ta", "ta-hi"]
INDIC_INDIC_PAIRS = {"hi-ta", "ta-hi"}


FLORES_DIR = os.path.join(ROOT, "flores200_dataset")
FLORES_URL = "https://tinyurl.com/flores200dataset"
FLORES_ALT_URL = "https://dl.fbaipublicfiles.com/flores200/flores200_dataset.tar.gz"


def _download_flores():
    """Download and extract FLORES-200 dataset if not already present."""
    import tarfile
    import urllib.request

    devtest_dir = os.path.join(FLORES_DIR, "devtest")
    if os.path.isdir(devtest_dir) and os.listdir(devtest_dir):
        return devtest_dir

    os.makedirs(FLORES_DIR, exist_ok=True)
    tar_path = os.path.join(FLORES_DIR, "flores200.tar.gz")

    for url in [FLORES_URL, FLORES_ALT_URL]:
        try:
            print(f"  Downloading from {url} ...")
            urllib.request.urlretrieve(url, tar_path)
            if os.path.getsize(tar_path) > 1_000_000:
                break
        except Exception as e:
            print(f"  Failed: {e}")

    if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 1_000_000:
        raise RuntimeError("Failed to download FLORES-200. Try manually:\n"
                           f"  bash -c 'cd {FLORES_DIR} && wget {FLORES_ALT_URL} -O flores200.tar.gz && tar xzf flores200.tar.gz'")

    print("  Extracting ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(FLORES_DIR)

    nested = os.path.join(FLORES_DIR, "flores200_dataset")
    if os.path.isdir(nested):
        for item in os.listdir(nested):
            src = os.path.join(nested, item)
            dst = os.path.join(FLORES_DIR, item)
            if not os.path.exists(dst):
                os.rename(src, dst)

    os.remove(tar_path)

    if not os.path.isdir(devtest_dir):
        raise RuntimeError(f"Extraction failed -- expected {devtest_dir}")

    return devtest_dir


def load_flores_devtest(languages: list[str]) -> dict[str, list[str]]:
    """Download FLORES-200 devtest and return {flores_code: [sentences]}."""
    flores_codes = [FLORES_LANG_MAP[lang] for lang in languages]
    print(f"[FLORES-200] Loading devtest for: {', '.join(flores_codes)} ...")

    devtest_dir = _download_flores()

    data = {}
    for code in flores_codes:
        fpath = os.path.join(devtest_dir, f"{code}.devtest")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing {fpath}. Available: {os.listdir(devtest_dir)[:10]}")
        with open(fpath, "r", encoding="utf-8") as f:
            data[code] = [line.strip() for line in f if line.strip()]
        print(f"  {code}: {len(data[code])} sentences")

    return data


def translate_sarvam(translator, sentences: list[str], source_lang: str, target_lang: str) -> tuple[list[str], float]:
    """Translate sentences one by one with Sarvam. Returns (hypotheses, total_seconds)."""
    hypotheses = []
    total_time = 0.0
    for i, sent in enumerate(sentences):
        result = translator.translate(sent, target_lang=target_lang, source_lang=source_lang)
        hypotheses.append(result["translation"])
        total_time += result["time_seconds"]
        if (i + 1) % 50 == 0 or (i + 1) == len(sentences):
            print(f"    Sarvam: {i + 1}/{len(sentences)} "
                  f"({total_time:.1f}s elapsed)")
    return hypotheses, total_time


def translate_indictrans(translator, sentences: list[str], source_lang: str, target_lang: str) -> tuple[list[str], float]:
    """Translate sentences one by one with IndicTrans2. Returns (hypotheses, total_seconds)."""
    hypotheses = []
    total_time = 0.0
    for i, sent in enumerate(sentences):
        result = translator.translate(sent, source_lang=source_lang, target_lang=target_lang)
        hypotheses.append(result["translation"])
        total_time += result["time_seconds"]
        if (i + 1) % 50 == 0 or (i + 1) == len(sentences):
            print(f"    IndicTrans2: {i + 1}/{len(sentences)} "
                  f"({total_time:.1f}s elapsed)")
    return hypotheses, total_time


def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    import sacrebleu
    return round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 2)


def compute_chrf(hypotheses: list[str], references: list[str]) -> float:
    import sacrebleu
    return round(sacrebleu.corpus_chrf(hypotheses, [references]).score, 2)


def compute_comet(sources: list[str], hypotheses: list[str], references: list[str], comet_model) -> float:
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
    output = comet_model.predict(data, batch_size=16, gpus=0)
    return round(output.system_score, 4)


def save_tsv(pair_name: str, model_name: str, sources: list[str], hypotheses: list[str], references: list[str]):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{pair_name}_{model_name}.tsv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("source\thypothesis\treference\n")
        for s, h, r in zip(sources, hypotheses, references):
            f.write(f"{s}\t{h}\t{r}\n")


def main():
    parser = argparse.ArgumentParser(
        description="FLORES-200 devtest benchmark for Sarvam-Translate vs IndicTrans2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pairs", nargs="+", default=ALL_PAIRS,
        help=f"Language pairs to benchmark (default: all). Choices: {ALL_PAIRS}",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N sentences (for quick testing)",
    )
    parser.add_argument(
        "--skip-comet", action="store_true",
        help="Skip COMET scoring (faster, no model download)",
    )
    parser.add_argument(
        "--sarvam-only", action="store_true",
        help="Only benchmark Sarvam-Translate",
    )
    args = parser.parse_args()

    for p in args.pairs:
        if p not in ALL_PAIRS:
            print(f"Unknown pair: {p}. Choices: {ALL_PAIRS}")
            sys.exit(1)

    needed_langs = set()
    for p in args.pairs:
        src, tgt = p.split("-")
        needed_langs.add(src)
        needed_langs.add(tgt)

    flores = load_flores_devtest(sorted(needed_langs))

    sarvam_translator = None
    indic_translator = None

    print("\n[Models] Loading Sarvam-Translate ...")
    from translate_sarvam import SarvamTranslator
    sarvam_translator = SarvamTranslator()

    if not args.sarvam_only:
        print("[Models] Loading IndicTrans2 ...")
        from translate_indictrans import IndicTransTranslator
        indic_translator = IndicTransTranslator()

    comet_model = None
    if not args.skip_comet:
        print("\n[COMET] Loading model: Unbabel/eamt22-cometinho-da ...")
        try:
            from comet import download_model, load_from_checkpoint
            model_path = download_model("Unbabel/eamt22-cometinho-da")
            comet_model = load_from_checkpoint(model_path)
            print("  COMET model loaded.")
        except Exception as e:
            print(f"  COMET failed to load: {e}")
            print("  Continuing without COMET. Install: pip install unbabel-comet")

    limit = args.limit
    width = 70

    all_results = []

    print(f"\n{'=' * width}")
    n_label = f"{limit}" if limit else "1012"
    print(f"  FLORES-200 Devtest Benchmark ({n_label} sentences per pair)")
    print(f"{'=' * width}")

    for pair in args.pairs:
        src_short, tgt_short = pair.split("-")
        src_flores = FLORES_LANG_MAP[src_short]
        tgt_flores = FLORES_LANG_MAP[tgt_short]

        sources = flores[src_flores][:limit]
        references = flores[tgt_flores][:limit]
        n = len(sources)

        is_indic_indic = pair in INDIC_INDIC_PAIRS
        run_indictrans = (not args.sarvam_only) and (not is_indic_indic) and (indic_translator is not None)

        pair_label = f"{src_short} → {tgt_short}"
        note = ""
        if is_indic_indic:
            note = "  (Sarvam via en pivot -- GGUF can't do direct Indic→Indic)"
        print(f"\n  {pair_label}{note}")
        print(f"  {'-' * (len(pair_label) + len(note))}")

        print(f"  Translating {n} sentences with Sarvam-Translate ...")
        sarvam_hyps, sarvam_time = translate_sarvam(sarvam_translator, sources, src_short, tgt_short)
        save_tsv(pair, "sarvam", sources, sarvam_hyps, references)

        indic_hyps, indic_time = None, None
        if run_indictrans:
            print(f"  Translating {n} sentences with IndicTrans2 ...")
            indic_hyps, indic_time = translate_indictrans(
                indic_translator, sources, src_short, tgt_short
            )
            save_tsv(pair, "indictrans", sources, indic_hyps, references)

        print(f"  Computing metrics ...")
        s_bleu = compute_bleu(sarvam_hyps, references)
        s_chrf = compute_chrf(sarvam_hyps, references)
        s_comet = None
        if comet_model:
            print(f"    COMET (Sarvam) ...")
            s_comet = compute_comet(sources, sarvam_hyps, references, comet_model)

        pair_result = {
            "pair": pair_label,
            "sarvam": {"bleu": s_bleu, "chrf": s_chrf, "comet": s_comet, "time": round(sarvam_time, 1)},
        }

        if run_indictrans:
            i_bleu = compute_bleu(indic_hyps, references)
            i_chrf = compute_chrf(indic_hyps, references)
            i_comet = None
            if comet_model:
                print(f"    COMET (IndicTrans2) ...")
                i_comet = compute_comet(sources, indic_hyps, references, comet_model)
            pair_result["indictrans"] = {"bleu": i_bleu, "chrf": i_chrf, "comet": i_comet, "time": round(indic_time, 1)}

        all_results.append(pair_result)

        rows = [
            ["BLEU", pair_result["sarvam"]["bleu"]],
            ["chrF++", pair_result["sarvam"]["chrf"]],
        ]
        if s_comet is not None:
            rows.append(["COMET", pair_result["sarvam"]["comet"]])
        rows.append(["Time (s)", pair_result["sarvam"]["time"]])

        headers = ["Metric", "Sarvam-Translate"]

        if "indictrans" in pair_result:
            for row in rows:
                metric = row[0]
                if metric == "BLEU":
                    row.append(pair_result["indictrans"]["bleu"])
                elif metric == "chrF++":
                    row.append(pair_result["indictrans"]["chrf"])
                elif metric == "COMET":
                    row.append(pair_result["indictrans"]["comet"])
                elif metric == "Time (s)":
                    row.append(pair_result["indictrans"]["time"])
            headers.append("IndicTrans2")

        print()
        print(tabulate(rows, headers=headers, tablefmt="grid", numalign="right"))

    results_path = os.path.join(RESULTS_DIR, "summary.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {RESULTS_DIR}/")

    print(f"\n{'=' * width}")
    print("  Summary")
    print(f"{'=' * width}")

    summary_rows = []
    for r in all_results:
        row = [r["pair"], r["sarvam"]["bleu"], r["sarvam"]["chrf"]]
        if r["sarvam"]["comet"] is not None:
            row.append(r["sarvam"]["comet"])
        else:
            row.append("--")
        if "indictrans" in r:
            row.extend([r["indictrans"]["bleu"], r["indictrans"]["chrf"]])
            if r["indictrans"]["comet"] is not None:
                row.append(r["indictrans"]["comet"])
            else:
                row.append("--")
        else:
            row.extend(["--", "--", "--"])
        summary_rows.append(row)

    summary_headers = [
        "Pair",
        "Sarvam BLEU", "Sarvam chrF++", "Sarvam COMET",
        "IT2 BLEU", "IT2 chrF++", "IT2 COMET",
    ]
    print()
    print(tabulate(summary_rows, headers=summary_headers, tablefmt="grid", numalign="right"))
    print(f"\n{'=' * width}\n")


if __name__ == "__main__":
    main()
