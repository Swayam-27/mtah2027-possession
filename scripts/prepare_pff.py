"""Convert PFF FC's 2022 World Cup data in external/pff to cache/PFF-*.npz and describe it.

Expects external/pff/{tracking,events,metadata}/ with one file per match id (the free dataset
needs registration with PFF FC, so it is not downloaded by a script). Matches missing either
the tracking or the event file are skipped.

    python scripts/prepare_pff.py
"""

import glob
import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
warnings.filterwarnings("ignore")
from possession import io_pff  # noqa: E402
from possession.match import Match  # noqa: E402
from prepare_metrica import describe  # noqa: E402

SOURCE = os.path.join(ROOT, "external", "pff")
CACHE = os.path.join(ROOT, "cache")

if __name__ == '__main__':
    tracking = {os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(SOURCE, "tracking", "*.jsonl.bz2"))}
    events = {os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(SOURCE, "events", "*.json"))}
    ids = sorted(tracking & events, key=int)
    print("{} matches with tracking and events ({} tracking, {} event files)".format(
        len(ids), len(tracking), len(events)), flush=True)
    for match_id in ids:
        path = os.path.join(CACHE, "PFF-{}.npz".format(match_id))
        started = time.time()
        if os.path.isfile(path):
            match = Match.load(path)
        else:
            try:
                match = io_pff.load(SOURCE, match_id)
            except Exception as error:  # a broken file should not stop the others
                print("PFF-{}: skipped ({})".format(match_id, error), flush=True)
                continue
            match.save(path)
        print(describe(match), " ({:.0f} s)".format(time.time() - started), flush=True)
