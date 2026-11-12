# Technical document — Round 1, ball possession

Team: Chandal Chowkdi · Draft. Numbers marked *external benchmark* come from other matches, not
from the competition match.

## 1. Approach

For every frame we output `H`, `A` or `D` inside the two halves and `X` outside them. The
pipeline has four stages:

1. **Features** from the tactical tracking: ball path with gaps filled by straight lines, ball
   speed, distance from the ball to the nearest player of each team, players near the ball,
   how far the ball is outside the pitch lines, and collective player movement (median and
   90th-percentile speed, share of players running, spread, drift of the centroid). Every player
   quantity is reduced over a team, never tied to a track slot, because slots are not people.
2. **Rule engine.** Dead ball only on positive evidence (ball seen off the pitch, ball
   motionless, players all walking, ball unseen for a long time). A team takes possession only
   by showing control: about a second on the ball, or two spells on the ball in a row. A lone
   touch changes nothing; the change is dated to the first touch.
3. **Tree model (fallback).** A gradient-boosted classifier (LightGBM, 3 classes) on the features, their
   values a few frames before and after, and averages over the preceding and following
   0.5 to 8 seconds. The rule engine's verdict is one of its inputs. Predictions are made twice,
   as given and with the teams swapped, and averaged, so the model cannot favour one side.
4. **Sequence model (submitted).** A small temporal convolutional network (MS-TCN style:
   3 stages of 10 dilated 1-D convolution layers, 64 channels, about 0.5 M parameters) that
   reads the per-frame features of the whole half and outputs H/A/D probabilities per frame.
   Trained with cross-entropy plus a smoothness term, on GPU, on the external matches only.
   Besides the ball and collective player features it reads 34 team-shape inputs, 17 per
   team, each measured in that team's own direction of attack (found per half from where
   its deepest player stands): how far up the pitch the team is, its length and width, its
   defensive line, whether it is advancing, how many of its players are ahead of the ball.
   They tell the network who has the ball when the ball itself is unseen. Three networks
   that differ only in their random seed are averaged, and every prediction is also averaged
   over the two team orderings.
5. **Decoding.** Viterbi over the per-frame probabilities with a fixed price for a change of
   team and a higher one for a dead/alive change. This removes flicker, which the ranking
   metric punishes as invented possessions.

Why this: on the external benchmark (leave-one-match-out over 7 matches, degradation
calibrated to summary statistics of the competition tracking, including the displacement of
airborne balls measured by comparing the two camera feeds), the sequence network scores
spellF1 0.74 with the team-shape inputs (worst match 0.69, best 0.77) and 0.69 without
them; it is better with them on all 7 matches. Adding the 3 Metrica matches to training raises the score on
the 7 held-out DFL matches from 0.745 to 0.754, and on the held-out Metrica matches from 0.773
to 0.789; the submitted networks are trained on all 10. Under changed degradation it stays at 0.72 to
0.75 (ball unseen on 55% or 25% of frames, no camera effect, twice the position error, short
gaps with jitter), where the version without team shape falls to 0.63 to 0.71. Under an
earlier, cruder degradation the baseline scored 0.30, the rule engine 0.50, the tree model
0.68 and the sequence model 0.78; the ordering of the methods is what we rely on from those.

An independent check on the real match itself, against the Smart Tagging (automatic, not
ground truth): after the tagged corners, goal kicks and free kicks, the team we give the
first possession to is the tagged team 23 times out of 24 (15 of 24 before the team-shape
inputs), and 55 of the 58 tagged restarts have one of our dead-ball stretches ending there.

## 2. Assets used and dropped

| Asset | Use |
|---|---|
| Tactical ATD (players + ball) | Used. The only input to the submitted pipeline |
| Smart Tagging | Not an input. Used as an independent check of the output on the real match (restart timing, team during tagged phases) |
| Broadcast ATD | Not an input. Aligned to the tactical timeline automatically and used to learn how the ball position errs (airborne balls), which calibrates the degradation of the training matches |
| Homography, tactical video, broadcast video | Not used |

## 3. Training data

No label of the competition match was used for training or tuning, and no human-produced data
about the competition match (event data, corrected tracking, annotations) was used at all. The
models learn only from other, public matches, which the competition rules allow:

- **DFL / Sportec open dataset** (Bassek et al., *Scientific Data* 2025, CC-BY 4.0): 7
  Bundesliga matches with corrected tracking at 25 fps. Its per-frame ball status and ball
  possession fields, produced by the data provider, give the `H`/`A`/`D` labels. Downloaded by
  `scripts/download_external.py` from the public mirror `huggingface.co/datasets/pysport/idsse-data`.
- **Metrica Sports sample data** (https://github.com/metrica-sports/sample-data, public,
  anonymised): 3 matches with tracking at 25 fps and synchronised event data. There are no
  per-frame labels, so they are built from the events by `possession/io_metrica.py`: dead from
  a stoppage event (ball out, at the moment it leaves the pitch; shot out or goal; foul;
  offside; hand ball) to the restart that follows, unless play went on (advantage); stretches
  of 2 s or more with no ball in the tracking are also dead (the data removes the ball exactly
  when play stops); otherwise the team of the last on-the-ball event (pass, carry, set piece,
  recovery, shot, the touch that lost the ball). Checks: the team nearest the ball agrees with
  these labels on 94-95% of frames where a player is within 1 m of it (DFL: 89-93%), and
  possession counts and lengths are close to the DFL matches. Downloaded by
  `scripts/download_metrica.py`.

Two adaptations make those matches resemble the task:

- **Labels.** DFL changes possession on every touch. Following the competition's definition
  (controlled possession; a loose ball stays with the last team), an opposing spell shorter than
  2 seconds is handed back to the team that had the ball. The same rule is applied to the
  Metrica labels.
- **Tracking.** The clean tracking is degraded to look like the competition's automatic
  tracking, with every setting calibrated against summary statistics of that tracking
  (`scripts/atd_stats.py`, `scripts/sim_stats.py`): the ball unseen on about 41% of frames in
  long stretches, short losses filled with straight lines, smoothed positions, a slowly
  wandering position error, airborne balls drawn on the ground where a single camera's line of
  sight lands (and lost when very high), most of the ball's trips beyond the lines unseen, and
  long player dropouts. During stoppages DFL freezes the ball at one point; this is replaced by
  a simulated ball that rolls out, disappears, is carried to the restart spot and placed. Each
  training match is degraded with many random noise settings around the calibrated one.

The public 5 minutes of ground truth were used only to check the pipeline (frame alignment,
label meaning), not to choose parameters.

## 4. Possession rules implemented

- **Loose and contested balls** stay with the last team in possession.
- **Change of possession** needs control, not a touch (see the 2-second rule above and the rule
  engine's control test).
- **Restarts.** The first team on the ball after a dead ball has possession from the restart.
- **Dead ball** is never inferred from the ball being unseen alone.

## 5. Limitations

- The external labels come from a different provider with a different possession definition;
  the 2-second conversion is our reading of the rules, not a measured equivalence.
- The noise model is a simulation. How well it matches the competition tracking decides how
  much of the benchmark score carries over.
- The model finds fewer possessions than exist (about 240 against 290 per match on the
  benchmark): short possessions are the main miss.
- Smart Tagging and the broadcast feed are unused.

## 6. Reproduction

See `README.md`: install pinned dependencies, place the data bundle, run
`python scripts/run_round1.py --out "outputs/Chandal Chowkdi.csv"`.
