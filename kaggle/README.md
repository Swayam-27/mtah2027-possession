# Training the sequence model on Kaggle

Kaggle gives a free GPU, which this laptop does not have. Only the open DFL matches are used
there. **The competition match data must never be uploaded to Kaggle.**

## Steps

1. Sign in at https://www.kaggle.com (a phone-verified account is needed for GPU and internet).
2. **Create → New Notebook**, then **File → Import Notebook** and upload
   `kaggle/train_sequence_model.ipynb` from this repo.
3. In the right-hand panel under *Session options* set:
   - **Accelerator:** GPU (T4 or P100)
   - **Internet:** On
4. Click **Run All**. It downloads the DFL matches (2.6 GB), converts them, runs a
   leave-one-match-out test and then trains the final model. Expect 1 to 2 hours.
   For a run that keeps going with the browser closed, use **Save Version → Save & Run All**.
5. When it finishes:
   - Copy the text from `mean spellF1 over 7 held-out matches` down to the end.
   - Download `sequence.pt` from the **Output** panel. If the browser saves it as
     `sequence.zip`, just rename it back to `sequence.pt` (do not unzip it).
6. Put `sequence.pt` into the `models/` folder of the project and share the copied scores.

## What the scores mean

The lines to look at are `sequence viterbi ...` under `mean spellF1`: scores on matches the
model never saw, damaged to look like the real tracking. The block after it shows the same
models under other kinds of damage, to check the score does not depend on one noise guess.
Differences under about 0.04 are within the scorer's noise.

## Rebuilding the notebook

The notebook contains a copy of the code. After changing anything in `possession/` or
`scripts/train_seq.py`, rebuild it:

```
python scripts/build_kaggle_notebook.py
```
