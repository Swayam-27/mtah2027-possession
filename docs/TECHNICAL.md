# Chandal Chowkdi — Round 1 technical document

**Team:** Chandal Chowkdi — Swayam Prajapati, Viranshul Panchal, Arya Patel, Yagnik Trivedi
**Task:** frame-by-frame ball possession (`H` / `A` / `D` / `X`) for Barcelona vs Athletic Club
**Code:** https://github.com/Swayam-27/mtah2027-possession

Every score in this document comes from **other matches** (public datasets, degraded to look like
the competition tracking, each scored by models that never saw it). Nothing was measured or tuned
on the competition match; its public 5 minutes were used only to check the pipeline.

## 1. Approach

**In one paragraph.** We train a temporal convolutional network on 74 public matches from three
providers, after damaging their clean tracking so that it looks like the competition's automatic
tracking. The network reads the whole half as a sequence of per-frame measurements — the ball
where it is seen, every team's shape and movement, and how both evolve over the surrounding
tens of seconds — and outputs `H`/`A`/`D` probabilities for every frame. A Viterbi decoder
turns those into possessions with no flicker. Twelve such networks are averaged.

**Why a sequence model.** The ball is unseen on 41% of graded frames, in stretches of ten seconds
and more, and those are often the moments that matter. A frame on its own frequently does not
contain the answer; the seconds around it do. The ranking metric also counts whole possessions,
so an output that flickers between teams is punished far more than one that is a little late.
A network that reads long context and a decoder that charges for every change address both.

**The pipeline.**

1. **Measurements per frame** (`possession/features.py`, `possession/shape.py`). Ball: position
   with gaps filled by straight lines, how long it has been unseen, speed and change of
   direction, distance to the nearest player of each team, players within 3 m, distance beyond
   the pitch lines. Players: speed distribution, share running, spread, drift of the centroid.
   **Team shape**, measured for each team in its own direction of attack (found per half from
   where its deepest player stands): depth on the pitch, length and width, the defensive line
   and its movement, whether the team is advancing, how many of its players are ahead of the
   ball, how close its second and third players are to the ball. Track slots in the competition
   tracking are not people, so every player quantity is a summary over a team, never a slot.
   The verdict of a hand-written rule engine (`possession/engine.py`) is also an input.
2. **Network** (`possession/seqmodel.py`). MS-TCN style: three stages of dilated 1-D
   convolutions (10 layers, 64 channels; the wider variant 11 layers, 96 channels), each later
   stage re-reading the previous stage's output. About ±40 s of context per frame. Every
   prediction is made twice, with the teams as given and swapped, and averaged, so the model
   cannot favour a side.
3. **Ensemble.** Twelve networks trained on all 74 matches: three training recipes (standard
   loss; weaker smoothness penalty; frames within 1 s of a change of label weighted three
   times) × three random seeds, plus three wider networks. Their probabilities are averaged.
4. **Decoding** (`possession/decode.py`). Viterbi over the probabilities with a fixed price for a
   change of team (12) and a higher one for a change between alive and dead (32), chosen on
   held-out external matches.
5. **Output.** `X` outside the two halves (published boundaries); every graded frame gets
   `H`, `A` or `D` — we never abstain inside a half.

**Results on held-out external matches** (spellF1, viterbi 12/32; each match scored by
networks trained without it; DFL is closest to how the competition's answers are made):

| Step | DFL (7 matches) | Metrica (3) |
|---|---|---|
| Organisers' baseline method | 0.32 | — |
| Rule engine alone | 0.46 | — |
| Network, no team-shape inputs | 0.69 | — |
| + team-shape inputs | 0.74 | — |
| + 3 Metrica matches in training | 0.75 | 0.79 |
| + 64 PFF World Cup matches | 0.77 | 0.82 |
| + twice the training length | **0.78** | 0.82 |
| bigger network, twice the length | 0.78 | **0.83** |
| four times the length | 0.79 | 0.83 |

The submitted twelve-network average was not itself scored on held-out matches (that would need
it retrained once per match); averaging versions has raised the held-out score every time we
measured it (e.g. three recipes: 0.757 vs 0.753 on DFL, better on 8–9 of 10 matches).
Four times the training length was not adopted: +0.002 on DFL, better on only 3 of 7 DFL
matches, i.e. the gain from longer training has levelled off.

**Robustness.** The damage applied to the training matches is our estimate of the competition
tracking, so every version was also scored under deliberately wrong estimates: the ball hidden
on 55% or 25% of frames, twice the position error, no airborne-ball effect, or the old
short-gap/jitter noise. Earlier versions held up under all of these; the PFF-trained versions
were tested with the ball hidden on 55% of frames and under the old noise, and stay at
0.76–0.82 on DFL and Metrica.

**Checks on the real match** (no answers involved): against the Smart Tagging (automatic, not
ground truth), 53 of the 58 tagged restarts have one of our dead-ball stretches ending within
3 s of them,
and after tagged corners, goal kicks and free kicks we give the ball to the tagged team 23
times out of 24. Our possessions on the full match average 15.4 s, against 14.2 s in the
public 5 minutes.

## 2. Assets used and dropped

| Asset | Use |
|---|---|
| **Tactical ATD** (players + ball) | **The only input of the submitted pipeline.** Its summary statistics (never its rows) also calibrated the damage applied to the training matches: share and length of ball gaps, smoothing, position error, player dropouts |
| **Broadcast ATD** | Not an input. Aligned to the tactical timeline automatically (players then agree within 0.3 m); the two ball feeds disagree, across the pitch and by metres, exactly when the ball is in the air. That taught us to simulate airborne balls drawn where a single camera's line of sight meets the ground. Merging it in would have filled only 9% of the missing ball, so it was dropped as an input |
| **Smart Tagging** | Not an input (no equivalent exists for the training matches). Used as an independent check of restarts and of the team in tagged phases (above) |
| Homography, tactical video, broadcast video | Not used. Ball detection from video was considered and dropped: the ball is missing mostly during stoppages and in the air, where a new detector would also fail, for weeks of work |
| Public 5-minute ground truth | Pipeline check only (frame alignment, label meaning). Never used to choose anything |

Also built and dropped: the rule engine alone (0.46), a gradient-boosted tree model on windowed
features (0.68 under an earlier noise model; kept in the repository as `--method model`),
mirrored-pitch test-time averaging (no gain), 75 decoder settings (none reliably better).

## 3. Training data

No label, event or annotation of the competition match was used. All training and validation
labels are on other games, as the rules allow:

| Dataset | Matches | Tracking | Labels: source and who made them |
|---|---|---|---|
| **DFL / Sportec open data** — Bassek et al., *Scientific Data* 2025, CC-BY 4.0; mirror `huggingface.co/datasets/pysport/idsse-data` | 7 Bundesliga | Corrected optical tracking, 25 fps | Per-frame ball status and ball possession produced by the league's official data provider (Sportec Solutions / DFL) |
| **Metrica Sports sample data** — github.com/metrica-sports/sample-data | 3 (anonymised) | Optical tracking, 25 fps | Built by our code (`possession/io_metrica.py`) from Metrica Sports' event data, tagged by their analysts |
| **PFF FC 2022 World Cup data** — free on registration with PFF FC | 64 World Cup matches | Broadcast tracking, 29.97 fps resampled to 25, off-screen players estimated by PFF | Built by our code (`possession/io_pff.py`) from PFF FC's event data, tagged by their analysts |

About 115 hours of football in total.

**Turning events into per-frame labels** (Metrica, PFF). Dead from each stoppage (ball out at the
moment it leaves the pitch, goal, foul, offside, hand ball — or, for PFF, the end of the last
on-the-ball event before a set piece) to the restart that follows, unless play went on
(advantage). Otherwise the team of the last on-the-ball event, carried forward. Checks: the team
nearest the ball agrees with these labels on 90–95% of frames where a player is within 1 m of it
(DFL's own labels: 89–93%); PFF possessions average 14.0 s, Metrica 12.5–13.5 s, against 14.2 s
in the competition's public 5 minutes.

**Matching the competition's definition.** DFL and the event-based labels change possession on
every touch. Following the challenge (controlled possession; a loose ball stays with the last
team), any opposing spell shorter than 2 s is handed back to the team that had the ball.

**Matching the competition's tracking.** Clean tracking is degraded before training, every
setting calibrated against summary statistics of the competition tracking
(`scripts/atd_stats.py`): ball unseen on ~41% of frames in long stretches, more often in the air,
in crowds and at stoppages; short losses filled with straight lines; smoothed positions; slowly
wandering ball error; airborne balls displaced as a single camera sees them and lost when very
high; most of the ball's trips beyond the lines unseen; long player dropouts. DFL freezes the
ball during stoppages, which would have taught the model a cue the competition data does not
have, so stoppages are rebuilt: the ball rolls out, disappears, is carried to the restart spot
and placed. Each training match is degraded several times with random settings around the
calibrated ones.

## 4. Possession rules implemented

- **Loose and contested balls** stay with the last team in possession; a lone touch,
  deflection or headed duel does not change it. A change needs control: in the training labels,
  at least 2 s with the ball or the start of a new spell that lasts.
- **Challenges** count for the winner only once the winner keeps the ball.
- **Restarts**: the team taking the restart has the ball from the restart.
- **Dead ball** is learned from the training labels and their evidence (ball beyond a line,
  players walking, a ball placed and still, a restart-like movement). There is no rule that an
  unseen ball is dead: the ball is unseen on 41% of frames, most of them in play.
- **Changes are dated to the moment control starts**, not when it is confirmed; the decoder's
  switching prices remove possessions too short to be real.

## 5. Limitations

- **The score is measured on other matches.** The damage applied to them is calibrated, and
  stress-tested, but it is still a simulation; the real score will differ. On the same damaged
  matches the organisers' baseline scores 0.32, against 0.24 on the real match.
- **Definitions differ by provider.** Labels from three providers, converted with our 2-second
  rule; the competition's exact cut-off for "control" is not published.
- **Short possessions are the main miss**: we find about 10% fewer possessions than exist.
- **Goal kicks** are the restart we place least precisely on the real match (8 of 10 within
  3 s of the tagged restart).
- **With more time:** fine-tune on the real tracking with self-generated labels, add the
  Smart Tagging as a learned input (needs equivalent tags on training data), and detect the
  ball in the tactical video where the tracking loses it.

## 6. Reproduction

Python 3.10. From a clone of the repository:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt      # Windows
python scripts/run_round1.py --data <folder with tactical/> --out "outputs/Chandal Chowkdi.csv"
python official/evaluation/validate_submission.py "outputs/Chandal Chowkdi.csv"
```

The trained networks are in `models/sequence.pt`, so no training is needed to reproduce the file;
a fresh clone with a new environment produced the identical file (byte for byte) in our test.
Retraining uses the open DFL and Metrica downloads (scripts provided), PFF's data (registration
needed, converted with `scripts/prepare_pff.py`) and a GPU: `kaggle/README.md` and
`scripts/train_seq.py`.
