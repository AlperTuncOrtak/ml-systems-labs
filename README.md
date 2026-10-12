# ML Systems Labs

Individual lab work for the ML Systems course (textbook: V. J. Reddi, *Introduction to Machine Learning Systems*).

## Setup

Python 3.11.8 is required (pinned in `.python-version`).

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Structure

Each lab has its own folder with `report.md`, `src/` (code) and `results/` (outputs). Datasets are downloaded into `data/` (git-ignored).

| Lab | Topic |
|---|---|
| [lab01](lab01/report.md) | Environment and first system measurements |
