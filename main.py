"""
Main workflow for the pharmacy adherence project.

This script loads raw prescription data, cleans it, saves the cleaned dataset,
generates exploratory visualizations, summarizes one patient's prescription
history, and trains simple linear and logistic regression models.
"""

from src.pharma_adherence.data import PharmaDataset
#TODO: Import ModelTrainer class from modeling.py
from src.pharma_adherence.modeling import ModelTrainer

"""
DAY 1: DATA PREPROCESSING & ANALYSIS
"""
# Load the raw prescription dataset into a PharmaDataset object.
# This class stores the dataframe and provides helper methods for cleaning,
# saving, visualization, and patient-level summaries.
dataset = PharmaDataset("data/raw/prescriptions_large_raw.csv")
print(dataset.df.head())
# Print the first few rows to confirm the file loaded correctly.
#TODO: Clean the dataset
dataset.clean()
# Clean the dataset by standardizing columns, fixing invalid values,
# and removing unusable rows.
#TODO: Save the dataset as a csv into "data/processed/"

# Save the cleaned dataframe so it can be reused without repeating preprocessing.
dataset.save("data/processed/prescriptions_large_cleaned.csv")
#TODO: Visualize the cleaned data
# Generate exploratory plots to understand drug distributions,
# adherence patterns, and age-related trends.
dataset.hist("drug_name").show()
dataset.bar("drug_name", "proportion_days_covered").show()
dataset.scatter("patient_age", "proportion_days_covered").show()

#TODO: Look at the summary of a patient
# Summarize one patient's prescription history.
patient = dataset.get_patient("P057")
print(patient.summary())
"""
DAY 2: MACHINE LEARNING
"""

#TODO: Instantiate a linear regression trainer
# Train a linear regression model to predict continuous PDC values.
linear_trainer = ModelTrainer(dataset.df, "proportion_days_covered", ["sex", "copay_amount"])

#TODO: Train the linear model
model, metrics = linear_trainer.train_linear()
#TODO: Print the linear model metrics
print("Linear regression metrics:")
print(metrics)
#TODO: Instantiate a logistic regression trainer
# Train a logistic regression model to predict binary adherence status.
logistic_model = ModelTrainer(dataset.df, "adherence_flag", ["sex", "copay_amount"])
#TODO: Train the logistic model
model, metrics = logistic_model.train_logistic()
#TODO: Print the logistic model metrics
print("Logistic regression metrics:")
print(metrics)