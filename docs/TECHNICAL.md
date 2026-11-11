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
   Predictions are averaged over the two team orderings.
5. **Decoding.** Viterbi over the per-frame probabilities with a fixed price for a change of
   team and a higher one for a dead/alive change. This removes flicker, which the ranking
   metric punishes as invented possessions.

Why this rather than rules alone: on the external benchmark (leave-one-match-out over 7
matches) the baseline scores spellF1 0.30, the rule engine 0.50, the tree model 0.68 and the
sequence model 0.78 (worst match 0.73, best 0.84).

## 2. Assets used and dropped

| Asset | Use |
|---|---|
| Tactical ATD (players + ball) | Used. The only input to the submitted pipeline |
| Smart Tagging | Not used in the current version |
| Homography, tactical video, broadcast video and ATD | Not used |

## 3. Training data

No label of the competition match was used for training or tuning. The model is trained on the
open **DFL / Sportec dataset** (Bassek et al., *Scientific Data* 2025, CC-BY 4.0): 7 Bundesliga
matches with corrected tracking at 25 fps. Its per-frame ball status and ball possession fields,
produced by the data provider, give the `H`/`A`/`D` labels.

Two adaptations make those matches resemble the task:

- **Labels.** DFL changes possession on every touch. Following the competition's definition
  (controlled possession; a loose ball stays with the last team), an opposing spell shorter than
  2 seconds is handed back to the team that had the ball.
- **Tracking.** The clean tracking is degraded: ball hidden on 30 to 50% of frames (more often
  in the air, in crowds and during stoppages), jitter, false ball detections, player dropouts.
  During stoppages DFL freezes the ball at one point; this is replaced by a simulated ball that
  rolls out, disappears, is carried to the restart spot and placed. Each training match is
  degraded with several random noise settings.

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
