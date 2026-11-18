"""Check how an EPTS metadata file lays out the player groups of each tracking line.

EPTS files can change the order of player groups part-way through (DataFormatSpecification
blocks with frame ranges). The competition loader assumes group i is track slot i on every
frame; this prints whether that holds. Structure only: counts and True/False, no data.

    python scripts/check_atd_layout.py
"""

import os
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession.io_atd import data_dir  # noqa: E402


def layout(meta_path):
    root = ET.parse(meta_path).getroot()
    channels = [c.get("id") for c in root.iter("PlayerChannel") if c.get("channelId") == "x"]
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        xs = [r for r in refs if r.endswith("_x")]
        specs.append((spec.get("startFrame"), spec.get("endFrame"), xs))
    return channels, specs


if __name__ == '__main__':
    folder = data_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    for source in ("tactical", "broadcast"):
        path = os.path.join(folder, source, "BAR-ATH_{}_atd.xml".format(source))
        if not os.path.isfile(path):
            continue
        channels, specs = layout(path)
        identity = [s[2] == channels[:len(s[2])] for s in specs]
        print("{}: {} player channels, {} format blocks, groups per block {}, "
              "every block in channel order: {}".format(
                  source, len(channels), len(specs), sorted({len(s[2]) for s in specs}),
                  all(identity)))
