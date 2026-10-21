# OCSMST

Official repository for the paper:

> **Behave Like Experts: Fusing Data with Semantic Knowledge for Effective and Efficient Semantic Map Construction**  
> Zhu Liu, Zhen Hu, Yuxuan Liu, Ying Liu  
> arXiv: [2610.25122](https://arxiv.org/abs/2610.25122)

This repository releases the **OCSMST benchmark dataset** and the **code** used in our paper, including the interactive semantic map construction web tool and the full experiment pipeline (LLM embedding similarity, fusion evaluation, and ablation studies).

## Repository Structure

```
OCSMST/
├── DATASET_EN/          # Benchmark dataset (English), 10 Excel files
├── DATASET_ZH/          # Benchmark dataset (Chinese), 12 Excel files
├── CODE/
│   ├── app.py           # Flask web application (interactive semantic map tool)
│   ├── SMM.py           # Semantic map model: candidate generation, fusion, evaluation
│   ├── index.html       # Frontend entry
│   ├── script.js        # Frontend logic (graph rendering & editing)
│   ├── styles.css       # Frontend styles
│   ├── front/           # User documentation (EN / ZH)
│   ├── libs/            # Frontend third-party libraries
│   ├── images/, logo/   # Assets
│   ├── data_pure/       # Case data used by the web tool / experiments (case1–case12)
│   ├── requirements.txt # Python dependencies
│   └── ARR_Oct/         # Experiment pipeline for the paper
│       ├── generate_LLM_contexts.py   # Build LLM prompts from case data
│       ├── compute_similarity_whole.py# Compute embedding similarities (multi-model)
│       ├── LLM_contexts/              # Generated prompts (case*.json)
│       ├── LLM_similarity/            # Similarity matrices & heatmaps per model
│       ├── main_evaluation/           # Main fusion evaluation & reports
│       ├── abalation_embed/           # Ablation: embedding similarity variants
│       ├── abalation_embed_norm/      # Ablation: normalized embedding fusion
│       ├── abalation_lambda/          # Ablation: fusion weight (λ) sensitivity
│       └── gt_matrix.csv              # Ground-truth matrices
└── README.md
```

## Dataset

### `DATASET_EN/` (main release)

Contains **10 Excel files** (`1.xlsx` – `10.xlsx`), one per case study. Every file has three sheets:

| Sheet | Description |
| --- | --- |
| **Form-function Table** | Rows are languages / dialects (or time periods and proficiency levels); columns are functions. A marked cell means the form realizes that function. |
| **GT** | The gold-standard function-to-function relation matrix (expert-annotated semantic map). |
| **Examples** | For each function: English name, Chinese name, explanation, and example sentences. |

Some files provide additional diachronic (`Sheet4`) and proficiency-level (`Sheet5`) breakdowns.

### `DATASET_ZH/` (extended Chinese version)

Contains **12 case files** (`case1.xlsx` – `case12.xlsx`) used in the experiments (superset of the 10 English cases). The same copies consumed by the web tool and evaluation scripts live in `CODE/data_pure/`.

## Code

### Web Tool (interactive semantic map construction)

Built on top of [XISM](https://github.com/hank317/XISM) (Apache-2.0). Users upload a form–function Excel table; the tool generates candidate semantic maps by fusing co-occurrence (connectivity constraint) with semantic similarity, supports interactive editing, quality evaluation, and PNG export.

```bash
cd CODE
pip install -r requirements.txt
python app.py            # then open http://localhost:5086 (port set in utils.py)
```

Excel format requirements are documented in `CODE/front/english_description.md` (and the in-app help / download guide).

### Experiment Pipeline (`CODE/ARR_Oct/`)

The paper's method fuses distributional semantic knowledge (from multilingual language models) into a top-down, data-driven SMM construction algorithm. The pipeline is:

1. **Prepare prompts** — extract per-function definitions and example sentences into LLM prompts:

   ```bash
   cd CODE/ARR_Oct
   python generate_LLM_contexts.py
   ```

2. **Compute embedding similarities** — encode functions with multiple embedding models (e.g., Qwen3-Embedding, mmBERT, XLM-R, bge-m3) and compute pairwise similarities per case:

   ```bash
   python compute_similarity_whole.py --cases 1,2,3 --models Qwen3-Embedding-8B
   ```

3. **Run fusion evaluation** — `main_evaluation/` runs the main experiments (NoBlend / Jaccard / Dice / Embeds fusion, Top-1 / Top-5 / Top-10 candidate selection) against the gold-standard maps and writes CSV reports and markdown summaries. The `abalation_*` directories hold the ablation studies on embedding variants, normalization, and the fusion weight λ.

Intermediate outputs (similarity matrices, heatmaps, logs) are kept in the respective subdirectories for reproducibility; detailed settings and results are summarized in each directory's `evaluation_report.md`.

## Citation

If you use this dataset or code, please cite:

```bibtex
@misc{liu2026behavelikeexpertsfusing,
      title={Behave Like Experts: Fusing Data with Semantic Knowledge for Effective and Efficient Semantic Map Construction},
      author={Zhu Liu and Zhen Hu and Yuxuan Liu and Ying Liu},
      year={2026},
      eprint={2610.25122},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.25122},
}
```

## License

- The web tool is adapted from [XISM](https://github.com/hank317/XISM), licensed under [Apache-2.0](./CODE/LICENSE).
- Please refer to the paper for details on the dataset and methodology.
