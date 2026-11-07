# Kàwé evaluation kit (Nigerian documents)

This kit produces every number in the paper's results section, using only Nigerian sources. Everything runs on your own computer from the `petty_talk` folder, in the Python environment you use for the app. It reads your Modal, ElevenLabs and YarnGPT keys from `.env`.

Every step saves as it goes. If a step stops (network drop, laptop sleeps), run the same command again and it carries on where it stopped, without paying twice for Modal calls.

**Overview of your part**

| Step | What you do | Your time |
|---|---|---|
| 1 | Collect 20–30 Nigerian documents | 2–3 hours |
| 2 | Collect 8 sentences each in Yoruba, Igbo, Hausa | 30 minutes |
| 3 | Run the scripts | about 2 hours, mostly waiting |
| 4 | Send rating packs to native speakers, collect them back | a few days |
| 5 | Run the analysis and send me `eval/results/` | 5 minutes |

---

## Step 0: one-time setup

```powershell
cd C:\Users\Benny\Desktop\petty_talk
# activate the environment you run the app with, e.g. .\.venv2\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r eval/requirements-eval.txt
```

Make sure your Modal app is deployed (`modal deploy modal_app.py`).

## Step 1: collect Nigerian documents

Create the folder `eval\nigerian_docs\` and put **20–30** documents in it (30 is better). Each document must meet all of these rules:

1. **Nigerian.** It must be published by a Nigerian government body, organization, university or journal. Leave out anything co-published with international bodies (World Bank, UN agencies, foreign universities).
2. **It has its own summary written by its authors.** That means an **Executive Summary** (reports) or an **Abstract** (theses, journal articles). That summary is what Kàwé's output is compared against.
3. **It's a text PDF or .docx, not a scan.** Check by opening the PDF: if you can highlight words with the mouse, it's fine.
4. **It's long enough:** at least about 8 pages of main text.
5. **It's in English.**

Aim for a mix, roughly half each:

- **Reports with an Executive Summary.** Examples of where to look:
  - National Bureau of Statistics (survey and statistical reports)
  - Central Bank of Nigeria (annual and economic reports)
  - NEITI industry reports
  - the Nigerian Economic Summit Group (macroeconomic outlooks)
  - the Nigeria Centre for Disease Control (annual reports)
  - Nigerian civil-society groups such as BudgIT or PLAC
  - federal or state planning documents
- **Academic work with an Abstract.** Examples:
  - theses and dissertations from Nigerian university repositories (UI, UNILAG, OAU, UNN, ABU and others)
  - final-year projects (with the author's permission)
  - articles from Nigerian journals; on African Journals Online you can filter by country = Nigeria

Then copy `eval\manifest_template.csv` to `eval\nigerian_docs\manifest.csv` and add one row per document. Open it in Excel:

| Column | What to put |
|---|---|
| file | exact file name, e.g. `nbs_labour_force_2023.pdf` |
| title, source, genre, year, url | for the paper's appendix; genre is `government report`, `thesis` or `journal article` |
| summary_pages | **leave empty** unless step 3a tells you otherwise |

The documents themselves stay on your computer: they are excluded from git.

## Step 2: collect native-language sentences

Open `eval\native_sentences\yor.txt`, `ibo.txt` and `hau.txt`. Add **8 or more** sentences to each, one per line. Each sentence should:

- come from a **Nigerian source** in that language, such as a Nigerian newspaper, radio or government website in Hausa, Yoruba or Igbo (for example Aminiya for Hausa, Alaroye for Yoruba), or be written by a native speaker you know;
- be a complete everyday sentence of 8–25 words with correct spelling and tone marks;
- be ordinary text, not headlines, lists or numbers.

Fill in the `# Source:` line at the top of each file with where the sentences came from.

These sentences are read aloud by the Yoruba, Igbo and Hausa voices. Because a native speaker wrote them, the ratings measure the voices themselves and not translation mistakes. The English sentences are taken from your documents automatically.

## Step 3: run the scripts

**3a. Prepare the test set** (a few seconds; no cost)

```powershell
python eval/prepare_corpus.py
```

Open `eval\results\corpus_review.md`. For every document it shows the first and last 15 words of the summary it found.

- If those look like the real start and end of the Abstract or Executive Summary, you're done.
- If not, or the document is listed under **Problems**, open the PDF, note the page numbers of its summary, put them in that row's `summary_pages` column (e.g. `3-5`) and run the command again. For Word files, use the "Section" numbers shown in the review.

**3b. Generate and score summaries** (about 30–60 minutes)

```powershell
python eval/run_summary_eval.py --percentage 30
python eval/score_summaries.py
```

The first script summarizes every document with Kàwé, your original university method (TextRank) and a simple "first N words" method. The second script scores them, measures speed, and checks with a separate AI checker whether each summary sentence is supported by the page it cites. The checker downloads about 0.7 GB the first time.

**3c. Translate summaries** (about 30–60 minutes)

```powershell
python eval/run_translation_eval.py --sections 30
```

This translates 30 Kàwé summary sections into Yoruba, Igbo and Hausa twice: once with Kàwé's translator and once with NLLB-200, a dedicated translation model, for comparison. NLLB runs on your laptop and downloads about 2.5 GB the first time.

**3d. Make the rating packs** (about 20–40 minutes)

```powershell
python eval/make_listening_test.py --sentences 8
```

This creates `eval\listening_test\` with folders `yor`, `ibo`, `hau` and `eng`. Add `--skip-english` if you don't want to spend ElevenLabs credits.

For each language it tests up to 4 YarnGPT voices (change with `--max-voices`). If YarnGPT's voice list says which languages each voice speaks, only matching voices are used; otherwise the 4 are picked at random from all voices, the same ones on every run. The voices used are printed at the start; note them for the paper.

## Step 4: native-speaker ratings

- **Who:** at least **5 native speakers per language** (3 is the bare minimum), plus 5 people for English. Classmates, colleagues, family, church or language-department contacts are all fine. Nobody who worked on Kàwé.
- **What to send:** zip each language folder (e.g. `yor`) and send it by WhatsApp or email. It already contains `RATER_INSTRUCTIONS.md`, the audio clips and the sheets. Raters need a laptop or a phone with a spreadsheet app and headphones; it takes them 25–30 minutes.
- **Never send `key.csv`.** It shows which voice and translator made each item, and raters must not know.
- **What comes back:** each rater returns `audio_ratings_XX.csv` and `translation_ratings_XX.csv`, where XX is their initials. Put them in the matching language folder, e.g. `eval\listening_test\yor\audio_ratings_AO.csv`.
- **For the paper,** keep a simple note of how many raters per language and how you found them. Ask each one whether they're happy to be thanked by name.

## Step 5: analyze and send me the results

```powershell
python eval/analyze_ratings.py
```

Then send me the whole `eval\results\` folder, plus your manifest and the `# Source:` lines. I'll put the numbers into the paper and write the results discussion, including anything that looks bad.

## Rough costs

- **Modal GPU time:** a few hours on an A10G in total, typically a few US dollars. Watch your Modal dashboard during step 3b.
- **ElevenLabs:** about 5,000 characters for the English clips.
- **YarnGPT credits:** about 100 short clips (4 voices × 8 sentences × 3 languages), roughly 10,000 characters in total. Check your balance on yarngpt.ai first.

## What each number means

| Measure | What it tells a reader |
|---|---|
| ROUGE-1/2/Lsum | Word overlap with the document's own abstract or executive summary. Higher is better; good for comparing systems, not an absolute quality score. |
| Supported by the cited passage | Share of summary sentences that the AI checker confirms follow from their cited page. This is the main evidence for "traceable to its source". |
| Supported by another section's passage | The same check against the wrong passage. This should be much lower; if it is, the citations are meaningful. |
| Naturalness / intelligibility (1–5) | Average native-speaker rating of each voice, with a 95% confidence range. |
| Adequacy / fluency (1–5) | Average native-speaker rating of each translator on real Kàwé summaries. |
