"""Build kaggle/train_sequence_model.ipynb: one self-contained notebook to train on a Kaggle GPU.

The notebook carries a copy of our code inside it, so it needs nothing but internet access to
download the open DFL matches. It never touches the competition data, which stays local.

    python scripts/build_kaggle_notebook.py
"""

import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "kaggle", "train_sequence_model.ipynb")
FILES = sorted(glob.glob(os.path.join(ROOT, "possession", "*.py"))) + \
    sorted(glob.glob(os.path.join(ROOT, "possession", "vendor", "*.py"))) + \
    [os.path.join(ROOT, "scripts", name) for name in
     ("download_external.py", "prepare_external.py", "download_metrica.py",
      "prepare_metrica.py", "train_seq.py")]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


if __name__ == '__main__':
    cells = [markdown(
        "# Train the possession sequence model\n\n"
        "Trains on the open DFL and Metrica sample matches only. **Do not upload the competition match data "
        "to Kaggle** -- it is not needed here and its terms forbid redistribution.\n\n"
        "Before running: in the right-hand panel set **Accelerator = GPU** and "
        "**Internet = On**. Then choose *Run All*. Takes roughly 3 hours, so use *Save Version -> Save & Run All*.\n\n"
        "When it finishes, download `sequence.pt` from the Output panel "
        "(`/kaggle/working/sequence.pt`) and copy the printed scores back to the team.")]

    cells.append(code("import os\n"
                      "os.makedirs('/kaggle/working/repo/possession/vendor', exist_ok=True)\n"
                      "os.makedirs('/kaggle/working/repo/scripts', exist_ok=True)\n"
                      "os.chdir('/kaggle/working/repo')\n"
                      "import torch; print('GPU available:', torch.cuda.is_available())"))
    for path in FILES:
        relative = os.path.relpath(path, ROOT).replace(os.sep, "/")
        with open(path, encoding="utf-8") as handle:
            cells.append(code("%%writefile {}\n{}".format(relative, handle.read())))

    cells.append(markdown("## 1. Download and convert the open DFL and Metrica matches (about 15 minutes)"))
    cells.append(code("!python scripts/download_external.py\n"
                      "!python scripts/prepare_external.py\n"
                      "!python scripts/download_metrica.py\n"
                      "!python scripts/prepare_metrica.py\n"
                      "!rm -rf external\n"
                      "# PFF World Cup matches, converted locally (scripts/prepare_pff.py) and attached\n"
                      "# to this notebook as a private Kaggle dataset; skipped if not attached.\n"
                      "!find /kaggle/input -name 'PFF-*.npz' -exec cp {} cache/ \\;\n"
                      "!ls cache | grep -c PFF || true"))
    cells.append(markdown("## 2. Train\n\nFirst a leave-one-match-out test (the printed "
                          "`mean spellF1` is the number that matters), "
                          "then the final model on all seven matches."))
    cells.append(code("!python scripts/train_seq.py --copies 8 --cv-steps 4000 --steps 8000 "
                      "--experiment pff --pff 64 --pff-copies 2 "
                      "--out /kaggle/working/sequence.pt"))
    cells.append(code("!ls -la /kaggle/working"))

    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                            "name": "python3"},
                             "language_info": {"name": "python"}}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(notebook, handle, indent=1)
    print("wrote {} ({} cells, {:.0f} KB)".format(OUT, len(cells), os.path.getsize(OUT) / 1024))
