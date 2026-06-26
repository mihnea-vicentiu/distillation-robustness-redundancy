# Distillation Robustness and Redundancy

This repository contains an academic experiment on robustness and redundancy in distilled encoder-decoder language models. The study explores how teacher-generated data, semantic similarity, and sample filtering can affect a smaller student model trained on instruction-style datasets.

The project can also serve as a practical reference for building information extraction and instruction-following pipelines, but its framing here is experimental: prepare public datasets, generate teacher outputs, train student models, curate examples, and analyze robustness through ablation notebooks.

## Repository Layout

```text
.
├── docs/
│   ├── references.md
│   ├── research-report.pdf
│   └── study-presentation.pdf
├── notebooks/
│   ├── analysis/
│   ├── experiments/
│   └── exploration/
├── src/distillation_robustness/
│   ├── curation/
│   ├── data/
│   ├── evaluation/
│   ├── teacher/
│   ├── training/
│   └── paths.py
├── LICENSE
└── requirements.txt
```

## Materials

- `docs/research-report.pdf` contains the written study material.
- `docs/study-presentation.pdf` contains the presentation material.
- `docs/references.md` tracks related work and future reading.
- `notebooks/exploration/` keeps early background work on distillation methods such as TinyBERT, MiniLM, and DistilBERT.
- `notebooks/experiments/` keeps experiment notebooks used to run model ablations.
- `notebooks/analysis/` keeps notebooks used to analyze ablation results and compare model behavior.

The draft notebooks are intentionally preserved because they document the ideas and intermediate steps that the final experiments build upon.

## Pipeline

Run commands from the repository root. The examples below use `PYTHONPATH=src` so the package can be imported without installing it.

1. Prepare Dolly data:

```bash
PYTHONPATH=src python -m distillation_robustness.data.prepare_dolly_dataset
```

2. Prepare Self-Instruct data:

```bash
PYTHONPATH=src python -m distillation_robustness.data.prepare_self_instruct_dataset
```

3. Generate teacher outputs for Self-Instruct train and validation splits:

```bash
PYTHONPATH=src python -m distillation_robustness.teacher.generate_teacher_outputs
```

4. Train a semantic-distillation student on the teacher outputs:

```bash
PYTHONPATH=src python -m distillation_robustness.training.train_semantic_student
```

5. Score teacher outputs against a smaller reference model:

```bash
PYTHONPATH=src python -m distillation_robustness.curation.score_teacher_outputs
```

6. Filter high-value examples:

```bash
PYTHONPATH=src python -m distillation_robustness.curation.filter_high_value_samples
```

7. Train a student on the filtered examples:

```bash
PYTHONPATH=src python -m distillation_robustness.training.train_filtered_student
```

8. Evaluate semantic similarity between teacher and student outputs:

```bash
PYTHONPATH=src python -m distillation_robustness.evaluation.evaluate_semantic_similarity
```

## Generated Artifacts

Generated datasets and checkpoints are intentionally excluded from version control. By default, scripts write to:

- `data/processed/` for prepared dataset splits.
- `data/teacher_outputs/` for teacher-labeled datasets.
- `data/scored/` for scored teacher outputs.
- `data/filtered/` for curated training subsets.
- `models/` for student checkpoints.

The evaluation script expects a Dolly teacher-output test dataset at `data/teacher_outputs/dolly/test` with `input` and `teacher_output` columns, and a student checkpoint at `models/dolly_filtered_student/final`.

## Setup

Create an environment with Python 3.10 or newer, then install dependencies:

```bash
pip install -r requirements.txt
```

The experiments use large language models and may require a CUDA-capable GPU with enough memory for teacher generation and student training.

## License

This project is released under the MIT License. See `LICENSE` for details.
