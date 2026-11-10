"""Download the seven open DFL / Sportec matches (other games, CC-BY 4.0) into external/dfl.

Source: the Hugging Face mirror of Bassek et al., "An integrated dataset of spatiotemporal and
event data in elite soccer", Scientific Data 2025. About 2.6 GB.

    python scripts/download_external.py
"""

import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "external", "dfl")
BASE = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
MATCHES = ("000001_DFL-MAT-J03WMX", "000001_DFL-MAT-J03WN1", "000002_DFL-MAT-J03WOH",
           "000002_DFL-MAT-J03WOY", "000002_DFL-MAT-J03WPY", "000002_DFL-MAT-J03WQQ",
           "000002_DFL-MAT-J03WR9")
KINDS = ("DFL_02_01_matchinformation", "DFL_04_03_positions_raw_observed")


if __name__ == '__main__':
    os.makedirs(TARGET, exist_ok=True)
    for match in MATCHES:
        for kind in KINDS:
            name = "{}_DFL-COM-{}.xml".format(kind, match)
            path = os.path.join(TARGET, name)
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                continue
            print("downloading {}".format(name), flush=True)
            partial = path + ".part"
            urllib.request.urlretrieve(BASE + name, partial)
            os.replace(partial, path)
    print("done: {}".format(TARGET))
    sys.exit(0)
