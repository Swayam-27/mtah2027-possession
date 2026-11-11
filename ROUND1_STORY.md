# Round 1 — the story so far

A plain-language record of what we are doing in Round 1 and why.
The newest progress is at the bottom, in the log.

## The job in one sentence

For every frame of the Barcelona vs Athletic Club tactical video, say who has the ball:
`H` (Barcelona), `A` (Athletic), `D` (ball not in play) or `X` (no answer).
One CSV, 199,319 rows. Deadline: **17 Nov 2026**. Top 20 teams go to Round 2.

## How we are judged

- The ranking number is **spellF1**. It counts how many real possessions we recovered as
  whole, correctly timed blocks. It does not reward getting most frames right.
- A possession is an unbroken run of `H` or of `A`. A dead ball (`D`) ends it.
  So `H H D D H H` is two possessions, not one.
- Being 1 second late everywhere turns a perfect answer into 0.65. Timing matters.
- Flickering labels (H, A, H, A...) create lots of fake short possessions and destroy the score.
- The score to beat is **0.2433** (the organisers' simple baseline).

## Why the simple baseline is weak

1. It says "dead ball" whenever the tracking cannot see the ball. The ball is unseen on 41% of
   scored frames, so it invents stoppages everywhere and cuts real possessions into pieces.
2. It gives the ball to whoever is nearest, so every duel or near-miss flips the team.
3. It smooths with a majority vote, which blurs the exact moment possession changes.

## Our plan

### Step 1 — Build our own practice exam
We get no leaderboard, and the 5 public minutes are too short to trust (and the rules forbid
tuning on them). So we use **other matches** that are public and already have the right answers:

- DFL / Sportec open data: 7 Bundesliga matches, players + ball at 25 fps.
- Metrica sample data: 3 games from the same company that made our data.

We then **damage these clean matches on purpose** so they look like ours: hide the ball on ~41%
of frames, break player tracks into pieces, remove player identities. Now we can test any idea
and get an honest score, as many times as we like.

### Step 2 — The rule-based engine (our safe submission)
A set of football rules written as code:

- **Rebuild the ball path** through the gaps where the tracking lost it.
- **Find touches**: moments where the ball changes speed or direction next to a player.
- **Change team only on real control**: two touches in a row by the new team, or about a second
  on the ball. A deflection or a lost header does not count. The change is dated back to the
  first touch, so we are not late.
- **Loose ball stays with the last team** (this is the organisers' own rule).
- **Dead ball only with evidence**: ball leaves the pitch, ball sits still at a restart spot,
  or the Smart Tagging marks a set piece. Never just because the ball is unseen.
- **Clean-up pass**: no possession shorter than a sensible minimum, no flicker.

### Step 3 — The learned model (our attempt to win)
Train a model on the damaged practice matches to make the two hardest calls better than fixed
rules can: "is the ball dead?" and "was that touch real control?". It replaces the rules **only
if it clearly beats them on practice matches it has never seen** (by more than 0.04, the
scorer's noise level).

### Step 4 — Recover the missing ball (only if the practice exam says it matters)
Use the second camera's tracking (broadcast) to fill ball gaps, after lining up the two timelines.

### Step 5 — Hand-in
- `<team_name>.csv`, checked with `validate_submission.py`.
- Public GitHub repo that rebuilds exactly the same CSV with one command. No match data in it.
- Short technical document.

## Rules we must not break

- No hand-labelling of our match. Labels come from code only.
- The 5 public minutes are for checking the pipeline works, never for tuning or training.
- Match data never goes to a cloud AI. So Claude writes the code and only ever sees summary
  numbers (scores, counts); a team member runs anything that touches the match data.
- Every external dataset we use is named in the technical document.

## Still open

- **Match data bundle is not on this computer yet** (searched D:, Downloads, Desktop, Documents).
  It has to be downloaded from the link in the enrolment email and unpacked into `official/data/`.
- Team name and members (3 to 5 people needed for Round 2).

## Progress log

- **10 Nov** — Read all official docs and the scoring code. Cloned the official repo into
  `official/`. Agreed the plan above. Wrote this file.
- **10 Nov** — Set up a Python 3.10 environment on D: (`.venv`). Ran the official scorer on the
  four shipped baselines: numbers match the README, so the scoring setup is right.
- **10 Nov** — Downloaded the 7 open DFL matches (other games, CC-BY licence) into `external/dfl`.
  Found one important difference: DFL flips possession on every touch (about 500 possessions per
  match, 37% shorter than 2 s), while our competition only counts *controlled* possession.
  Added a step that hands short "one-touch" spells (under 2 s) back to the team that had the
  ball, so the practice answers follow the competition's definition.
- **10 Nov** — Built the practice exam: clean DFL tracking is damaged on purpose (ball hidden on
  41% of frames, players dropping out, jitter, false ball detections).
- **10 Nov** — First scores on the practice exam (one match): organisers' baseline **0.26**
  (their real score on our match is 0.24, so the practice exam is realistic), our rule engine
  **0.43**. Checks show the bigger loss is in deciding *which team* has the ball, not in
  spotting dead balls.
- **10 Nov** — Wrote the learned model (LightGBM + Viterbi decoding) and the one-command
  submission script.
- **10 Nov** — Caught a trap: in the DFL data the ball is frozen at one spot during every
  stoppage, so the first model "detected" dead balls from a quirk our real data will not have
  (its first score of 0.66 was not trustworthy). Fixed the practice exam: during stoppages the
  ball now rolls out, disappears, is carried to the restart spot by a player and placed. Training
  matches are also damaged with many different random noise settings, so the model cannot rely
  on one guess about the noise being right.
- **10 Nov** — Full practice exam, all 7 matches, each one scored by a model that never saw it:

  | Method | spellF1 (mean of 7) | worst match | best match |
  |---|---|---|---|
  | Organisers' baseline | 0.30 | 0.26 | 0.33 |
  | Rule engine | 0.50 | 0.44 | 0.53 |
  | Learned model + decoding | **0.68** | 0.64 | 0.76 |

  The model also stays ahead under noise it never trained on (0.58 with 55% of the ball hidden;
  0.67 to 0.72 when the dead-ball picture is changed).
- **10 Nov** — Checked dead-ball detection using player movement only (no ball at all): still
  good (0.88). So most of that skill is real and does not depend on how we simulated the ball.
- **10 Nov** — Rehearsed the whole pipeline on a fake file in the competition's format: loader,
  engine, CSV writer and the official validator all pass.
- **10 Nov** — Tried two ideas to recover missed short possessions: "who was last / is next on
  the ball" features, and a second model that predicts where possession changes. Neither moved
  the score beyond noise (all between 0.65 and 0.68), so the second model was dropped and the
  simple decoder kept.
- **10 Nov** — Trained the final model on all 7 practice matches (`models/possession.txt`).
  Rehearsed the full pipeline with it on a fake file in the competition's format: it ran, the
  official validator accepted the CSV, and two runs produced the identical file.
- **10 Nov** — Put the code in a private GitHub repo (to be made public at submission time).
- **10 Nov** — Wrote a second, stronger kind of model (a small neural network that reads
  the match as a sequence) to try to beat 0.68. It needs a GPU, so it is trained on Kaggle
  using only the public DFL matches; see `kaggle/README.md`. The match data never goes to
  Kaggle. Not trained yet, so its score is unknown.
- **11 Nov** — First Kaggle run trained nothing: one notebook cell was empty and stopped the
  run. Fixed, rerun, and the trained file is now in `models/sequence.pt`. It loads and runs
  on the laptop (a full match in about 5 seconds). Its honest score, on matches it never
  saw, is printed in the Kaggle log and still has to be compared with the tree model's 0.68
  before we decide which one to submit.
- **11 Nov** — Kaggle scores are in. On matches it never saw, the sequence model scores
  **0.78** (worst match 0.73, best 0.84) against the tree model's 0.68 (its worst match is close
  to the tree model's best). It also finds about the right number of possessions, where
  the tree model found too few. It is now the default in `scripts/run_round1.py`; the tree
  model stays as a fallback (`--method model`). Not yet checked: how the sequence model
  holds up under noise it was not trained on.
- **Ready for the real data.** When it arrives: run `scripts/atd_stats.py`, share the summary
  numbers, adjust the practice exam's noise if needed, then run `scripts/run_round1.py`.
- **Still unknown until the real data arrives:** how closely the practice exam's noise matches
  the real tracking, and so how much of the 0.78 carries over.

## What is in the folder

| Path | What it is |
|---|---|
| `possession/` | Our code: loaders, features, rule engine, model, decoding, scoring |
| `scripts/prepare_external.py` | Converts downloaded DFL matches into fast cached files |
| `scripts/bench.py`, `scripts/cv.py` | The practice exam |
| `scripts/atd_stats.py` | Prints summary numbers about our match data (safe to share) |
| `scripts/run_round1.py` | One command: match data in, submission CSV out |
| `official/` | The organisers' repo (docs, scorer). Match data goes in `official/data/` |
| `external/`, `cache/`, `outputs/` | Downloads and generated files, never committed |
