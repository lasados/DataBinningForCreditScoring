# Data Binning For Credit Scoring

This is a Python package for solving binning problem in credit scoring.

[**This article**](https://www.researchgate.net/publication/322520135_Monotone_optimal_binning_algorithm_for_credit_risk_modeling) is the basis for creating the package.

Useful resources :
[article about WOE and IV](https://medium.com/@sundarstyles89/weight-of-evidence-and-information-value-using-python-6f05072e83eb)

## Installation
Clone repository in folder on your system.
```bash
git clone https://github.com/lasados/Data-Binning-for-credit-scoring.git
```
Open repository.

Install requirements with pip.
```bash
pip install -r pip-requirements.txt
```

## Usage
```python
from models.monotonic import *

data_train = pd.read_csv('your_data.csv')
# name_target_column = 'y'

df_woe_train, corr_matrix, to_drop, iv_values, full_cut_stats, use_name_iv = start_pipeline(data_train, 'compare')
```

See full pipeline in [notebooks/pipeline.ipynb](https://github.com/lasados/Data-Binning-for-credit-scoring/blob/master/notebooks/pipeline.ipynb)

Usage example - [notebooks/model.ipynb](https://github.com/lasados/Data-Binning-for-credit-scoring/blob/master/notebooks/model.ipynb)