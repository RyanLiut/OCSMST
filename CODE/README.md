# OCSMST Web Tool

Interactive semantic map construction tool used in our paper
([arXiv:2610.25122](https://arxiv.org/abs/2610.25122)).

Upload a form–function Excel table, generate candidate semantic maps by fusing
co-occurrence (connectivity constraint) with distributional semantic
similarity, edit the graph interactively, evaluate map quality, and export
PNG images.

The web frontend/backend is adapted from
[XISM](https://github.com/hank317/XISM) (Apache-2.0); see [LICENSE](./LICENSE).

## Quick Start

```bash
pip install -r requirements.txt
python app.py    # default port in utils.py (5086)
```

Then open `http://localhost:5086` in your browser.

## Layout

| Path | Description |
| --- | --- |
| `app.py` | Flask backend (`/api/process-excel`, merge, etc.) |
| `SMM.py` | Semantic map model: candidate trees, fusion, metrics |
| `index.html`, `script.js`, `styles.css` | Frontend |
| `front/` | In-app user documentation (EN / ZH) |
| `libs/`, `images/`, `logo/` | Static assets |
| `data_pure/` | Case Excel files used by the tool and experiments |
| `ARR_Oct/` | Experiment pipeline for the paper (see root README) |

Excel format requirements: see `front/english_description.md` or the in-app
help (Download Guide).
