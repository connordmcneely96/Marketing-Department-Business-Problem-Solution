# Notebooks

This directory contains Jupyter notebooks for exploratory data analysis and experimentation.

## Contents

- `project-notebook.ipynb` - Main notebook exported from Google Colab (to be added)

## Usage

When you export your notebook from Google Colab, place it here with a descriptive name.

The notebook can stay messy and exploratory - the production code goes in `src/`.

## Tips

1. Export from Colab: File → Download → Download .ipynb
2. Name it descriptively (e.g., `customer-churn-eda.ipynb`)
3. Keep data loading relative to repo root
4. Use `sys.path.append('..')` to import from `src/`

Example notebook cell to import from src:

```python
import sys
sys.path.append('..')

from src.config import RAW_DATA_FILE
from src.preprocess import load_raw_data
from src.train import train_models
```
