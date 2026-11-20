"""Runs the evaluation end to end. Every step resumes where it stopped if interrupted.

    python eval/run_all.py                     # everything
    python eval/run_all.py --skip-translation  # summaries, scoring and figures only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-translation", action="store_true")
    parser.add_argument("--skip-nli", action="store_true", help="skip the NLI faithfulness checks")
    args = parser.parse_args()

    nli = ["--skip-nli"] if args.skip_nli else []
    steps = [["prepare_corpus.py"], ["run_summaries.py"], ["score_summaries.py", *nli]]
    if not args.skip_translation:
        steps.append(["run_translation_eval.py", *nli])
    steps.append(["make_figures.py"])

    for step in steps:
        print(f"\n=== {' '.join(step)} ===", flush=True)
        start = time.perf_counter()
        result = subprocess.run([sys.executable, str(EVAL_DIR / step[0]), *step[1:]])
        if result.returncode != 0:
            sys.exit(f"{step[0]} failed; fix the error and run this command again to resume.")
        print(f"--- {step[0]} finished in {time.perf_counter() - start:.0f}s", flush=True)
    print("\nAll done. Results are in eval/results/ and figures in eval/results/figures/.")


if __name__ == "__main__":
    main()
