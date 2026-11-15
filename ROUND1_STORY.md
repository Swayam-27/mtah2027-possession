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
- `Chandal Chowkdi.csv`, checked with `validate_submission.py`.
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
- Team **Chandal Chowkdi**: Swayam Prajapati, Viranshul Panchal, Arya Patel, Yagnik Trivedi.

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
- **11 Nov** — The real tracking arrived (only the two tracking files are needed; videos not
  downloaded). Compared its summary numbers with our practice exam. The share of missing ball
  matched (41%), but the character of the damage did not:
  - The real feed is **smooth**, not jittery. I had been adding shake that is not there.
  - The ball goes missing in **long stretches** (typically 10 seconds, mostly during
    stoppages), not in short flickers. Short losses are filled in by the tracker with straight
    lines.
  - The ball is drawn **further from the players' feet** than in the practice matches, so
    "who is nearest" is less reliable than I assumed.
  - Players drop out for long spells, not short ones.
- **11 Nov** — Rebuilt the practice exam's damage to match all of the above, and checked the
  summary numbers of a damaged practice match now sit close to the real ones. The models
  trained on the old damage must be retrained on the new one (next Kaggle run). That run also
  scores each model under several other kinds of damage, so we see how fragile the score is.
- **11 Nov** — Second Kaggle run done: the sequence model retrained on the realistic damage
  is in `models/sequence.pt` (the tree model was retrained too, as the fallback). Wrote the
  submission `outputs/Chandal Chowkdi.csv` with it: the official validator accepts it and two
  runs give the identical file.
- **11 Nov** — Honest practice score of the retrained model, on matches it never saw:

  | Practice damage | spellF1 (mean of 7) | worst | best |
  |---|---|---|---|
  | Matched to the real tracking | **0.74** | 0.70 | 0.80 |
  | Ball hidden on 55% of frames instead of 41% | 0.66 | 0.61 | 0.73 |
  | Ball drawn twice as far off its true spot | 0.70 | 0.67 | 0.73 |
  | Ball drawn exactly on its true spot | 0.74 | 0.71 | 0.80 |
  | The old damage (short gaps, shaky positions) | 0.74 | 0.72 | 0.78 |

  Lower than the earlier 0.78 because the realistic damage is harder, but it holds up when
  the damage is changed. The decoder settings (12/32) were confirmed again.
- **11 Nov** — Third Kaggle run: tried averaging several models (three copies of the network
  with different random starts, plus one larger network). On unseen practice matches:
  single network 0.744, three averaged 0.746, larger network alone 0.733, all four averaged
  0.749. The gain is tiny, inside the scorer's noise, but the average was never worse under
  any of the changed-damage tests (for example 0.674 against 0.660 with 55% of the ball
  hidden), so the four-model average is now what `models/sequence.pt` holds and what the
  submission is made from. Lesson: a bigger or repeated model is not where more score is.
- **11 Nov** — Read the organisers' definition again before touching the "2-second rule" for
  short possessions. Their answers come mechanically from corrected tracking, like our
  practice matches, and their published average possession length (14.2 s on the public
  5 minutes) is close to ours on the real match (14.8 s). So the rule stays.
- **11 Nov** — Found a real gap in what the model is shown: it knew how all 22 players move
  together, but nothing about each team separately, and not even which way each team attacks.
  Added "team shape" inputs (for each team: how far up the pitch it is, how wide and long, is
  it advancing or dropping, where its back line is, how many players are ahead of the ball).
  On a practice match the team with the ball is clearly wider and moving forward.
- **11 Nov** — Second camera (broadcast tracking). Wrote `scripts/broadcast_sync.py`, which
  lines it up with the tactical tracking automatically (it found one time shift per half, and
  players then sit within 0.3 m of each other in the two feeds). Result: it would fill only
  about 9% of the frames where our ball is missing, so merging it in is not worth the risk.
  But it taught us something better: when the two cameras disagree about the ball, it is
  across the pitch and by metres. That is the mark of a ball in the air, which a single camera
  draws on the ground further away than it really is. The practice matches know the ball's
  real height, so this effect is now simulated in the practice exam (plus: very high balls
  are lost, and most of what the ball does beyond the lines is not seen). The damaged practice
  matches now match the real feed on every summary number we print.
- **11 Nov** — Smart Tagging used as an independent check on the real match
  (`scripts/tagging_check.py`), not as an input. Good news: 55 of 58 tagged restarts have one
  of our dead-ball stretches ending right there, and during tagged build-up and progression
  phases we give the ball to the tagged team 91-92% of the time. Bad news: after corners we
  give the first possession to the team taking it only 8 times in 12, and after goal kicks 5
  in 8. The model could not know better, because it did not know which way teams attack; the
  team-shape inputs should fix that. We will re-run this check on the next model.
- **12 Nov** — The team-shape run is in, and it is the biggest single gain since the
  sequence model itself. On unseen practice matches with the improved (harder) damage:

  | | with team shape | without |
  |---|---|---|
  | Matched to the real tracking | **0.743** | 0.688 |
  | Ball hidden on 55% of frames | 0.722 | 0.627 |
  | Ball hidden on 25% of frames | 0.747 | 0.708 |
  | Ball drawn with twice the position error | 0.737 | 0.668 |
  | No airborne-ball camera effect | 0.733 | 0.684 |
  | The old damage (short gaps, shaky positions) | 0.723 | 0.700 |

  Better on all 7 matches, and much steadier when the ball is hidden more. (The 0.743 is not
  comparable with yesterday's 0.749: the exam is harder now. Without team shape, the same
  network falls from 0.749 to 0.688 on the new exam.)
- **12 Nov** — Real-match check against the Smart Tagging: after tagged corners, goal kicks
  and free kicks, the team we give the ball to now matches the tag **22 times out of 23**
  (was 15 of 24). This is the first evidence from the real match itself, not from practice
  matches, and it says the improvement is real there too. The 5-minute pipeline check moved
  from 0.51 to 0.68 (still too small a sample to trust on its own).
- **12 Nov** — Submission regenerated with the three team-shape networks
  (`outputs/Chandal Chowkdi.csv`): official validator accepts it, two runs identical.
- **Still unknown:** the real score. The practice exam now resembles the real tracking much
  more closely, but it is still other matches with simulated damage.

- **12 Nov** — Added Metrica's three public sample games as more practice matches (the
  model has learned from only 7 so far). They have no frame-by-frame possession, so the
  answers are built from their event data: dead from a stoppage (ball out, foul, offside,
  goal) to the restart, and otherwise the team that last had the ball under control, with the
  same 2-second rule as before. Checks on the result: the team nearest the ball agrees with
  these answers 94-95% of the time (DFL: 89-93%), possession counts and lengths look like the
  DFL matches, and the ball vanishes from Metrica's tracking within one frame of every
  stoppage found in the events. Fixed one trap on the way: game 3 reorders its players at
  every substitution, which first scrambled the teams. Also confirmed our competition loader
  does not have that problem (one fixed layout for the whole match).
- **12 Nov** — Metrica run is in. Every one of the 10 matches held out in turn:

  | Held-out matches | trained on DFL only | DFL + Metrica |
  |---|---|---|
  | 7 DFL matches | 0.745 | **0.754** |
  | 3 Metrica matches | 0.773 | **0.789** |
  | DFL, ball hidden 55% | 0.723 | 0.738 |
  | DFL, old-style damage | 0.724 | 0.738 |

  A small gain (about +0.01, below the scorer's noise on its own), but it points the same way
  in every test, so the submission now uses the networks trained on all 10 matches. On the
  real match the Smart Tagging check gives 23 of 24 restarts to the tagged team. Submission
  regenerated, validated, reproducible.

- **12 Nov** — Tried two ways to stop the model merging short possessions: weaker smoothing
  in training, and making the frames around every change count three times more. On their own
  neither beats the current model on the DFL matches (all three 0.753). But the held-out
  predictions were saved this time, so I could test combinations on the laptop: **the average
  of the three versions** scores 0.757 on DFL and 0.800 on Metrica (current: 0.753 / 0.786),
  and is better on 8 or 9 of the 10 matches under every kind of damage tested. 75 decoder
  settings were also compared; none beats the current one reliably, so it stays.
- **12 Nov** — Goal kicks on the practice matches: the right team gets the restart 93.5% of the
  time, no worse than other restarts, so the practice matches do not show the goal-kick
  weakness seen on the real match. Nothing to fix there without real-match answers.
- **13 Nov** — The two extra versions are trained on all 10 matches. The submission now
  averages nine networks (three versions x three random starts). Regenerated, validated, two
  runs identical, 23 of 24 restarts agree with the Smart Tagging. Takes about 1.5 minutes.

- **13 Nov** — Added PFF FC's free 2022 World Cup data (64 matches, broadcast tracking plus
  detailed events; free registration). Answers built from the events like Metrica's; their
  average possession length (14.0 s) is the closest yet to the organisers' public 5 minutes
  (14.2 s). Result, every DFL and Metrica match held out in turn:

  | Held-out matches | trained without PFF | with PFF |
  |---|---|---|
  | 7 DFL matches | 0.753 | **0.768** |
  | 3 Metrica matches | 0.786 | **0.823** |
  | DFL, ball hidden 55% | 0.735 | 0.742 |
  | DFL, old-style damage | 0.737 | 0.764 |

  Better on all 10 matches. The biggest gain since the team-shape inputs. Mirrored
  predictions were tested in the same run and did not help, so they are not used.
- **13 Nov, evening — where things stand.** Three Kaggle runs queued overnight:
  1. `kaggle/train_sequence_model.ipynb`: the final nine networks with PFF. Its `sequence.pt`
     goes in `outputs/`; then regenerate the CSV, validate, run `scripts/tagging_check.py`.
  2. `kaggle/experiment_longer.ipynb`: same network trained twice as long.
  3. `kaggle/experiment_wide.ipynb`: bigger network trained twice as long.
  Runs 2 and 3 are compared with "with PFF" = 0.768 on DFL / 0.823 on Metrica. If one clearly
  wins, retrain the final nine that way. Then: update the technical document, re-run the
  fresh-clone reproduction check, make the repo public, submit when the form arrives (it has
  not arrived yet). PFF data: converted files are in a private Kaggle dataset
  `pff-wc22-converted`; local copies in `cache/PFF-*.npz`.

- **14 Nov** — Both overnight experiments won (every DFL and Metrica match held out):

  | | DFL | Metrica | DFL, ball hidden 55% |
  |---|---|---|---|
  | with PFF (4000 training steps) | 0.768 | 0.823 | 0.742 |
  | with PFF, twice as long | **0.784** | 0.822 | 0.756 |
  | with PFF, bigger network, twice as long | 0.780 | **0.830** | **0.765** |

  Longer training is better on 6 of 7 DFL matches. The final models were already trained for
  the longer length; the final file now also adds three of the bigger networks, since they
  hold up best when more of the ball is hidden.

- **14 Nov** — Four times the training length: 0.786 on DFL against 0.784 for twice the length,
  better on only 3 of 7 DFL matches. The gain has levelled off, so not adopted.
- **14 Nov** — **Final model installed**: twelve networks trained on all 74 matches (three
  recipes x three seeds, plus three bigger networks). Submission regenerated: official
  validator accepts it, two runs identical (about 2 minutes). Real-match checks: 23 of 24
  set-piece restarts go to the tagged team; 53 of 58 tagged restarts have one of our dead-ball
  stretches ending within 3 s; possessions average 15.4 s (public 5 minutes: 14.2 s).
- **15 Nov** — Video ball detection tried and dropped. A pretrained detector run locally on
  2,000 frames: after removing detections beyond the lines, filling 20% of the missing ball
  costs a 34% error rate (we required 80% accuracy before looking). And it could not matter much:
  on the real match the ball is hidden in only 8.6% of live play (83.8% of dead time), close to
  our simulation (3–10% and 84%), so the practice test does resemble the real match.
- **15 Nov** — Timing of possession changes checked on the practice matches: no early/late bias
  (mean 0.06 s, half within ±0.4 s); shifting them only lowers the score. Nothing left to fix
  there. Submission unchanged; to be submitted by the team on 16 Nov.

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
