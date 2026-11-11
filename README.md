# More than a Hack 2027 — Round 1: ball possession

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
the 199,319-row CSV.

## Retrain the model (optional)

The submitted sequence model was trained on a GPU with `scripts/train_seq.py`; see
`kaggle/README.md`. The steps below rebuild the tree model (`--method model`), the fallback.

The model is trained only on other matches: the open DFL / Sportec tracking dataset
(Bassek et al., *Scientific Data* 2025, CC-BY 4.0). To rebuild it from scratch, download the
seven matches into `external/dfl/`, then:

```
python scripts/prepare_external.py     # convert to cached arrays
python scripts/cv.py                   # leave-one-match-out test (optional)
python scripts/train_model.py          # writes models/possession.txt
```

## Layout

| Path | Purpose |
|---|---|
| `possession/io_atd.py`, `io_dfl.py` | Loaders into one common match format |
| `possession/features.py` | Per-frame measurements (ball path, distances, player movement) |
| `possession/engine.py` | Rule-based possession engine |
| `possession/model.py`, `decode.py` | Learned model and Viterbi decoding |
| `possession/degrade.py`, `relabel.py` | Make external matches resemble the competition data |
| `possession/vendor/` | The organisers' metric code, unchanged (MIT) |
| `scripts/` | Entry points |
