"""Print the summary statistics of one degraded external match (no competition data needed).

    python scripts/sim_stats.py [match index]
"""

import glob
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
from possession import relabel  # noqa: E402
from possession.degrade import degrade  # noqa: E402
from possession.diagnostics import print_stats, tracking_stats  # noqa: E402
from possession.match import Match  # noqa: E402

if __name__ == '__main__':
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    match = Match.load(sorted(glob.glob(os.path.join(ROOT, "cache", "DFL-*.npz")))[index])
    match.labels = relabel.controlled(match.labels, match.runs, 50)
    print_stats("DEGRADED EXTERNAL MATCH " + match.name, tracking_stats(degrade(match)))
