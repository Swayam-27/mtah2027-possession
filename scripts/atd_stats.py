"""Describe the competition tracking in summary numbers, next to a degraded external match.

Prints aggregates only (shares, percentiles, counts) -- no rows, frames or positions -- so the
output is safe to share when calibrating the benchmark's noise model.

    python scripts/atd_stats.py                       # data in official/data or MTAH_DATA_DIR
    python scripts/atd_stats.py --data D:\\mtah2027
"""

import argparse
import glob
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
from possession import io_atd  # noqa: E402
from possession.degrade import degrade  # noqa: E402
from possession.diagnostics import print_stats, tracking_stats  # noqa: E402
from possession.match import Match  # noqa: E402


def data_dir(explicit=None):
    return explicit or os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "official", "data")


def load_competition(folder):
    stem = os.path.join(folder, "tactical", "BAR-ATH_tactical")
    meta, raw = stem + "_atd.xml", stem + "_atd.txt"
    for path in (meta, raw):
        if not os.path.isfile(path):
            raise SystemExit("match data not found: {}\nUnpack the bundle into {} or pass --data."
                             .format(path, folder))
    return io_atd.load(meta, raw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="folder holding tactical/ and broadcast/")
    args = parser.parse_args()

    print_stats("COMPETITION TRACKING (BAR-ATH tactical)",
                tracking_stats(load_competition(data_dir(args.data))))

    cached = sorted(glob.glob(os.path.join(ROOT, "cache", "DFL-*.npz")))
    if cached:
        external = Match.load(cached[0])
        print_stats("DEGRADED EXTERNAL MATCH (what the benchmark currently simulates)",
                    tracking_stats(degrade(external)))
        print_stats("CLEAN EXTERNAL MATCH (before any degradation)", tracking_stats(external))
