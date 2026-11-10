"""Score a label track with the official metric code (vendored unchanged, MIT licence)."""

import numpy as np

from .match import X
from .vendor import possession_metrics as pm


def graded_truth(match):
    """Ground-truth codes with everything outside the match's play runs forced to X."""
    truth = np.full(match.n_frames, X, dtype=np.int8)
    for start, stop in match.runs:
        truth[start:stop] = match.labels[start:stop]
    return truth


def score(truth, pred):
    """The numbers we steer by. spell is the ranking metric."""
    result = pm.evaluate(truth, np.asarray(pred, dtype=np.int8))
    return {
        "spell": result["spell_f1_mean"],
        "macro": result["macro_f1"],
        "n_gt": result["n_gt_spells"],
        "n_pred": result["n_pred_spells"],
        "dur": result["mean_spell_duration_s"],
        "dur_gt": result["gt_mean_spell_duration_s"],
        "alive_dead": result["alive_dead_f1"],
        "team_acc": result["team_acc_both_alive"],
        "f1_D": result["per_class"]["D"]["f1"],
        "spell_by_iou": {k: v["f1"] for k, v in result["spell_f1"].items()},
    }


def line(name, s):
    return ("{:<28} spell {:.4f}  macro {:.4f}  poss {:>4}/{:<4} dur {:5.1f}/{:<5.1f} "
            "alive/dead {:.3f}  team {:.3f}  D-F1 {:.3f}").format(
                name, s["spell"], s["macro"], s["n_pred"], s["n_gt"], s["dur"], s["dur_gt"],
                s["alive_dead"], s["team_acc"], s["f1_D"])
