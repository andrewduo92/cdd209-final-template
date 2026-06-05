import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

#TODO: Will need more imports!
#      LinearRegression
#      Logistic Regression
#      various sklearn metrics
#      train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    roc_auc_score
)
from sklearn.model_selection import train_test_split


class ModelTrainer:
    """
    Train and evaluate machine learning models for the pharmacy adherence dataset.

    The class takes a cleaned dataframe, a target column, and selected feature
    columns. It builds preprocessing pipelines for numeric and categorical data,
    then trains either a linear regression model or logistic regression model.
    """

    def __init__(self, df: pd.DataFrame, target: str, features: list[str]):
        """
        Initialize the model trainer.

        Parameters:
            df: Cleaned dataframe used for modeling.
            target: Name of the outcome column to predict.
            features: List of feature columns used as predictors.
        """
        self.df = df
        self.target = target
        self.features = features
        self.model = None
        self.metrics = None

    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        DO NOT ALTER THIS METHOD!

        A preprocessing pipeline that automates the cleaning and formatting 
        of your data so it's ready for a machine learning model!

        Splits data into numeric features and categorical features.
        SimpleImputer fills any remaining missing data with column median.
        OneHotEncoder turns text categories into digit categories.
        """
        numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

    def train_linear(self):
        """
        Train and evaluate a linear regression model.

        This method is used for continuous outcomes such as
        proportion_days_covered. It splits the dataset into training and test
        sets, applies preprocessing, trains a linear regression model, and
        returns regression metrics.

        Returns:
            A fitted sklearn Pipeline and a dictionary containing MSE, MAE,
            and R-squared.
        """
        x = self.df[self.features]
        y = self.df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=57
        )

        preprocessor = self.build_preprocessor(X_train)

        self.model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        )

        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)

        self.metrics = {
            "mse": mean_squared_error(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
            "r2": r2_score(y_test, preds)
        }

        return self.model, self.metrics

    def train_logistic(self):
        """
        Train and evaluate a logistic regression model.

        This method is used for binary outcomes such as adherence_flag. It
        predicts whether a patient is adherent or non-adherent using the
        selected feature columns. The method returns common classification
        metrics including accuracy, precision, recall, and ROC-AUC.

        Returns:
            A fitted sklearn Pipeline and a dictionary containing accuracy,
            precision, recall, and ROC-AUC.
        """
        x = self.df[self.features]
        y = self.df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=57
        )

        preprocessor = self.build_preprocessor(X_train)

        self.model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        )

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs),
        }

        return self.model, self.metrics

    def evaluate(self):
        """
        Return the metrics from the most recently trained model.

        Raises:
            ValueError: If no model has been trained yet.

        Returns:
            Dictionary of model evaluation metrics.
        """
        if self.metrics is None:
            raise ValueError("No metrics available yet. Train a model first.")
        return self.metrics