# Pharma Adherence

Pharma Adherence is a Python project for cleaning, analyzing, visualizing, and modeling prescription adherence data. The project focuses on transforming raw prescription records into a cleaned dataset that can be used for exploratory data analysis and simple machine learning models. It includes tools for standardizing messy pharmacy data, generating patient-level adherence summaries, visualizing adherence-related patterns, and training regression models to evaluate medication adherence outcomes.

## Features

- Clean raw prescription data by standardizing column names, dates, numeric values, drug names, pharmacy names, and categorical fields
- Remove duplicate rows and rows missing critical fields such as patient ID, fill date, or drug name
- Save cleaned prescription data to a processed CSV file
- Generate visualizations of prescription and adherence patterns
- Create patient-level adherence summary profiles
- Calculate patient metrics such as total fills, average copay, average days supply, and proportion of days covered
- Train a linear regression model to predict continuous adherence outcomes
- Train a logistic regression model to predict binary adherence status
- Evaluate model performance using regression and classification metrics

## Project Structure

```text
cdd209-final-template/
├── data/
│   ├── raw/
│   │   └── prescriptions_large_raw.csv
│   └── processed/
│       └── prescriptions_large_cleaned.csv
├── src/
│   └── pharma_adherence/
│       ├── cleaning.py
│       ├── data.py
│       ├── modeling.py
│       ├── patient.py
│       └── visualization.py
├── tests/
├── main.py
├── pyproject.toml
└── README.md
```

## Installation

### Clone the Repository

```bash
git clone [URL]
cd cdd209-final-template
```

Replace `[URL]` with the GitHub repository URL.

### Create a Virtual Environment

Using conda:

```bash
conda create --name cdd209 python=3.13
conda activate cdd209
```

Alternatively, any Python environment with Python 3.10 or newer should work.

### Install the Package

From the root of the repository:

```bash
pip install .
```

For development and testing tools, install the optional development dependencies:

```bash
pip install .[dev]
```

## Usage

Run the main workflow from the root of the repository:

```bash
python main.py
```

The script performs the following steps:

1. Loads the raw prescription dataset from:

```text
data/raw/prescriptions_large_raw.csv
```

2. Cleans the dataset using the cleaning functions in `cleaning.py`.

3. Saves the cleaned dataset to:

```text
data/processed/prescriptions_large_cleaned.csv
```

4. Generates exploratory visualizations, including:

- Histogram of drug names
- Bar chart of average proportion of days covered by drug name
- Scatter plot of patient age versus proportion of days covered

5. Creates a patient adherence summary for patient `P057`.

6. Trains a linear regression model to predict `proportion_days_covered`.

7. Trains a logistic regression model to predict `adherence_flag`.

8. Prints model performance metrics to the terminal.

## Example Output

The patient summary returns a dictionary like this:

```python
{
    "patient_id": "P057",
    "total_fills": 1796,
    "avg_copay": 14.70,
    "avg_days_supply": 76.37,
    "pdc": 0.661,
    "adherent": False
}
```

The linear regression model reports metrics such as:

```python
{
    "mse": 0.012,
    "mae": 0.084,
    "r2": 0.21
}
```

The logistic regression model reports metrics such as:

```python
{
    "accuracy": 0.82,
    "precision": 0.79,
    "recall": 0.75,
    "roc_auc": 0.86
}
```

Exact values may differ depending on the dataset and preprocessing results.

## Testing

After installation, run the test suite from the repository root:

```bash
pytest
```

This will run the tests in the `tests/` folder and check that the package functions are working as expected.

## Main Modules

### `cleaning.py`

Contains helper functions for cleaning raw prescription data. It standardizes text, converts date and numeric fields, normalizes categorical values, handles missing values, and removes invalid records.

### `data.py`

Defines the `PharmaDataset` class, which stores the prescription dataframe and provides methods for cleaning, saving, visualizing, and retrieving patient-specific data.

### `patient.py`

Defines the `PatientAdherenceProfile` class, which summarizes one patient's prescription history and adherence behavior.

### `visualization.py`

Contains plotting functions for histograms, bar charts, and scatter plots.

### `modeling.py`

Defines the `ModelTrainer` class, which builds preprocessing pipelines and trains linear or logistic regression models.

## Requirements

This project requires Python 3.10 or newer. Main dependencies include:

- pandas
- numpy
- matplotlib
- scikit-learn
- scipy
- pytest
