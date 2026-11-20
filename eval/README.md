# Evaluation

The test set is every PDF or Word file in `eval/evaluation_papers/`. Each paper's own abstract is the reference summary; Kàwé summarizes the body (after the abstract, before the references). Outputs go to `eval/results/`, figures to `eval/results/figures/`.

## Setup

```powershell
python -m pip install -r requirements.txt -r eval/requirements-eval.txt
modal deploy modal_app.py
```

The scripts read the Modal URLs and tokens from `.env`. The whole-document baseline replaces the system prompt and sends a full paper, which the endpoints accept only with `EVAL_AUTH_TOKEN` (add it to the `rag-auth-token` Modal secret and to `.env`, or pass `--skip-document-baseline` to `run_summaries.py`).

Summary budgets are percentages of a paper's passages (`kawe.metrics.summary_top_n`), as in the app.

## Run

```powershell
python eval/run_all.py
```

This runs, in order (each step can also be run on its own and resumes if interrupted):

| Step | Script | Needs | Output |
|---|---|---|---|
| 1 | `prepare_corpus.py` | — | `corpus.jsonl`, `extraction.jsonl`, `corpus_review.md` |
| 2 | `run_summaries.py` | Modal | `summaries.jsonl`, `passage_cache.jsonl`, `embeddings/`, `cold_start.jsonl` |
| 3 | `score_summaries.py` | NLI model (CPU, ~0.7 GB download) | `scores.jsonl`, `positions.jsonl`, `grounding.jsonl`, `summary_report.md`, `summary_scores.json` |
| 4 | `run_translation_eval.py` | Modal, NLLB (CPU, ~2.5 GB download) | `translations.jsonl`, `translation_report.md`, `translation_scores.json` |
| 5 | `make_figures.py` | — | `figures/*.pdf`, `figures/*.png` |

Check `corpus_review.md` after step 1: each row shows where the detected abstract starts and ends.

## What is measured

| Analysis | Systems | Measure |
|---|---|---|
| Content overlap | Kàwé, TextRank, Lead at 10–50% budgets | ROUGE-1/2/Lsum F1 against the abstract |
| Ablations (30%) | Kàwé with random passages, with the first passages, and a whole-document LLM summary | ROUGE and faithfulness |
| Faithfulness | all abstractive systems | NLI entailment of each sentence by its cited passage, by another section's passage (control), and by the best of the 5 most similar passages |
| Abstractiveness | all systems | novel 1/2/3-grams relative to the body |
| Provenance | Kàwé, ablations, TextRank | position of summarized passages vs. the passages that best match the abstract |
| Efficiency | Kàwé, whole-document LLM | seconds per stage on a warm GPU; cold-start time |
| Extraction | PDFium vs. pdfplumber | words recovered and fused-word rate |
| Translation | Llama-3.1-8B vs. NLLB-200 | round-trip chrF and mutual entailment |

Intervals are 95% percentile bootstrap intervals over papers; paired comparisons report the mean difference, wins and the Wilcoxon signed-rank p-value. With seven papers, per-paper tests have limited power, so sentence-level faithfulness results carry most of the statistical weight.

## Native-speaker ratings (optional)

```powershell
python eval/make_listening_test.py --sentences 8
python eval/analyze_ratings.py
```

`make_listening_test.py` builds blinded rating packs in `eval/listening_test/<lang>/` from the native-written sentences in `eval/native_sentences/` and the translations from step 4. Keep `listening_test/key.csv` private. Raters return their sheets with their initials in the file name (for example `yor/audio_ratings_AO.csv`).
