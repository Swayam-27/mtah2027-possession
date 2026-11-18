# More than a Hack 2027 â€” Round 1: ball possession

Frame-by-frame ball possession (`H` / `A` / `D` / `X`) for Barcelona vs Athletic Club, from the
automatic tracking data provided by the organisers.

No match data is stored in this repository. `ROUND1_STORY.md` explains the approach in plain
language.

## Reproduce the submission

Python 3.10.

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt        # Windows
# .venv/bin/python -m pip install -r requirements.txt          # macOS / Linux
```

Unpack the competition data bundle so that `official/data/tactical/` exists, or pass its
location with `--data` (or set `MTAH_DATA_DIR`). Then:

```
python scripts/run_round1.py --out "outputs/Chandal Chowkdi.csv"
```

This reads the tactical tracking, applies the trained sequence model in `models/sequence.pt` and writes
the 199,319-row CSV. It also prints the file's SHA-256: the same data bundle and pinned
packages must give the same hash on every machine.

### The submitted file

| | |
|---|---|
| File | `Chandal Chowkdi.csv` (Round 1, submitted 17 Nov 2026) |
| SHA-256 | _to fill in: the hash `run_round1.py` prints_ |
| Public 5-minute window (pipeline check only) | spellF1 _to fill in_ · macroF1 _to fill in_ |
| Held-out external matches | spellF1 0.78 on DFL (7), 0.82 on Metrica (3), see `docs/TECHNICAL.md` |

`--round-prob` (off by default) adds a second rounding step for future submissions. It changes
the output, so never use it to reproduce the Round 1 file.

## Retrain the model (optional)

The submitted sequence model was trained on a GPU with `scripts/train_seq.py`; see
`kaggle/README.md`. The steps below rebuild the tree model (`--method model`), a **legacy**
fallback: it was scored under an older noise model (0.68 spellF1 on held-out DFL), so its numbers
cannot be compared with the sequence model's (0.78).

The model is trained only on other matches: the open DFL / Sportec tracking dataset
(Bassek et al., *Scientific Data* 2025, CC-BY 4.0). To rebuild it from scratch, download the
seven matches into `external/dfl/`, then:

```
python scripts/prepare_external.py     # convert to cached arrays
python scripts/cv.py                   # leave-one-match-out test (optional)
python scripts/train_model.py          # writes models/possession.txt
```

## Model files and Git LFS

`models/` is in plain git so a fresh clone reproduces the submission with no extra tools. Keep it
that way until the Round 1 results are out (before 26 Nov 2026), because the judges reproduce
from this repository. After that, move the models to LFS so each retrain stops growing the clone:

```
git lfs install
git lfs track "models/*.pt" "models/*.txt"     # appended after "* -text" in .gitattributes
git add .gitattributes models && git commit -m "Models in Git LFS"
```

`* -text` stays first in `.gitattributes` so every file is still byte-identical on every platform.

## Layout

| Path | Purpose |
|---|---|
| `possession/io_atd.py`, `io_dfl.py` | Loaders into one common match format (`io_atd.load_competition` finds the bundle) |
| `possession/features.py` | Per-frame measurements (ball path, distances, player movement) |
| `possession/engine.py` | Rule-based possession engine |
| `possession/model.py`, `decode.py` | Learned model and Viterbi decoding |
| `possession/degrade.py`, `relabel.py` | Make external matches resemble the competition data |
| `possession/vendor/` | The organisers' metric code, unchanged (MIT) |
| `scripts/` | Entry points |
