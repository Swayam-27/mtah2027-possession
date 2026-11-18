"""One command: competition data in, submission CSV out.

    python scripts/run_round1.py                          # sequence model (models/sequence.pt)
    python scripts/run_round1.py --method model           # legacy tree model (models/possession.txt)
    python scripts/run_round1.py --method rules           # rule engine only, no model needed
    python scripts/run_round1.py --method both            # average of tree and sequence models
    python scripts/run_round1.py --data D:\\mtah2027 --out outputs\\team.csv

Prints summary numbers only. If the public 5-minute ground truth is available it is scored as a
pipeline check (frames aligned, labels meaning what we think) -- never used to tune anything.
"""

import argparse
import hashlib
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
from possession.io_atd import data_dir, load_competition  # noqa: E402
from possession import decode, engine, features, model  # noqa: E402
from possession.match import D, LABELS, X  # noqa: E402
from possession.scoring import line, score  # noqa: E402
from possession.vendor import possession_metrics as pm  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "models", "possession.txt")
SEQUENCE_PATH = os.path.join(ROOT, "models", "sequence.pt")
PUBLIC_GT = os.path.join(ROOT, "official", "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
SWITCH_TEAM, SWITCH_DEAD = 12.0, 32.0
PREKICKOFF_DEAD = (110808, 113742)  # graded, all D, second half not yet kicked off (FAQ.md)
# Log-probabilities are rounded before Viterbi so tiny floating-point differences between
# machines (CPU instruction sets, thread counts) cannot change the decoded file.
ROUND_LOG = 3


def write_submission(path, codes):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, LABELS[c]) for i, c in enumerate(codes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--method", choices=("model", "rules", "sequence", "both"),
                        default="sequence",
                        help="sequence: the submitted model. model: legacy LightGBM tree model, "
                             "scored under an older noise model, not comparable to the sequence "
                             "model. rules: rule engine only. both: average of model and sequence")
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "submission.csv"))
    parser.add_argument("--sequence", default=SEQUENCE_PATH, help="sequence-model checkpoint")
    parser.add_argument("--round-prob", type=int, default=None, metavar="DECIMALS",
                        help="also round the averaged probabilities before decoding. Off by "
                             "default: switching it on changes the file, so it is for a new "
                             "submission only, never to reproduce the Round 1 one")
    args = parser.parse_args()

    match = load_competition(data_dir(args.data))
    f = features.compute(match)
    if args.method == "rules":
        codes = engine.label_match(match, f)
    else:
        probs = []
        if args.method in ("model", "both"):
            import lightgbm as lgb
            probs.append(model.predict(lgb.Booster(model_file=MODEL_PATH), match, f))
        if args.method in ("sequence", "both"):
            import torch
            from possession import seqmodel
            torch.set_num_threads(1)  # fixed summation order: same numbers on every machine
            members = seqmodel.load_all(args.sequence)
            probs.append(seqmodel.predict_all(members, match, f))
        prob = sum(probs) / len(probs)
        if args.round_prob is not None:
            prob = np.round(prob.astype(np.float32), args.round_prob)
        # Published by the organisers (FAQ, games.py): the second half is graded from 110,808
        # but kicks off at 113,742, and every frame in between is D. Not a hand label: a fact
        # of the scoring window, applied before decoding so the kickoff transition is clean.
        a, b = PREKICKOFF_DEAD
        unsure = int(np.sum(prob[a:b].argmax(axis=1) != D))
        prob[a:b] = (1e-6, 1e-6, 1.0)
        print("pre-kickoff window: model alone said not-D on {} of {} frames; forced to D"
              .format(unsure, b - a))
        codes = decode.decode_match(prob, match.runs, SWITCH_TEAM, SWITCH_DEAD,
                                    round_log=ROUND_LOG)

    assert len(codes) == match.n_frames
    write_submission(args.out, codes)

    graded = np.concatenate([np.arange(a, b) for a, b in match.runs])
    shares = {LABELS[k]: float(np.mean(codes[graded] == k)) for k in range(3)}
    stats = pm.possession_stats(codes, match.runs)
    print("wrote {}  ({} rows, method: {})".format(args.out, len(codes), args.method))
    with open(args.out, "rb") as handle:
        print("sha256 {}  (compare with the README to confirm an exact reproduction)"
              .format(hashlib.sha256(handle.read()).hexdigest()))
    print("graded frames: H {H:.3f}  A {A:.3f}  D {D:.3f}".format(**shares))
    print("possessions: {}  mean {:.1f} s".format(stats["n_spells"], stats["mean_duration_s"]))

    if os.path.isfile(PUBLIC_GT):
        truth = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
        print("pipeline check on the public 5 minutes (not a tuning signal):")
        print("  " + line(args.method, score(truth, codes)))
    print("\nNow run:  python official/evaluation/validate_submission.py \"{}\"".format(args.out))
