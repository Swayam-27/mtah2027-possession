"""Download Metrica Sports' three public sample games (other games, anonymised) into external/metrica.

Source: https://github.com/metrica-sports/sample-data -- tracking (25 fps, players and ball)
and synchronised event data. About 210 MB. Used only as extra training and test matches.

    python scripts/download_metrica.py
"""

import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "external", "metrica")
BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/"
FILES = (
    "data/Sample_Game_1/Sample_Game_1_RawEventsData.csv",
    "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
    "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
    "data/Sample_Game_2/Sample_Game_2_RawEventsData.csv",
    "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
    "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
    "data/Sample_Game_3/Sample_Game_3_events.json",
    "data/Sample_Game_3/Sample_Game_3_metadata.xml",
    "data/Sample_Game_3/Sample_Game_3_tracking.txt",
    "documentation/events-definitions.pdf",
)


if __name__ == '__main__':
    os.makedirs(TARGET, exist_ok=True)
    for name in FILES:
        path = os.path.join(TARGET, os.path.basename(name))
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            continue
        print("downloading {}".format(os.path.basename(name)), flush=True)
        partial = path + ".part"
        urllib.request.urlretrieve(BASE + name, partial)
        os.replace(partial, path)
    print("done: {}".format(TARGET))
    sys.exit(0)
