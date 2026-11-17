# Summary evaluation (19 Nigerian documents)

## Table 1: ROUGE F1 against human reference summaries (mean ± SD)

| System | ROUGE-1 | ROUGE-2 | ROUGE-Lsum | Words |
|---|---|---|---|---|
| Kàwé (grounded) | 42.0 ± 6.4 | 15.9 ± 4.3 | 39.1 ± 5.7 | 506.7 ± 202.3 |
| TextRank (thesis system) | 36.2 ± 7.7 | 15.5 ± 6.8 | 34.6 ± 7.2 | 521.3 ± 198.9 |
| Lead | 32.8 ± 6.5 | 10.6 ± 5.6 | 30.4 ± 6.1 | 506.7 ± 202.3 |


## Table 2: Kàwé processing time

| Measure | Value |
|---|---|
| Source length (words) | 5173.6 ± 3524.5 |
| Passages rewritten per document | 9.0 ± 3.3 |
| Summary time, warm GPU (s) | 58.6 ± 19.3 |
| Median summary time (s) | 59.4 |
| GPU cold start (s) | 28.8 ± 33.1 |
| Passages that failed and were skipped | 0 |
| Documents where the 15-passage cap applied | 2 |


## Table 3: Grounding of Kàwé summary sentences (NLI check)

| Measure | Value |
|---|---|
| Summary sentences checked | 462 |
| Supported by the cited passage | 77.7% |
| Contradicted by the cited passage | 6.5% |
| Supported by another section's passage (control) | 17.5% |
