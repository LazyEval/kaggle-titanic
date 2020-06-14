# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
from src.data import DataSet
from src.models import Model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = [LogisticRegression(), SVC(), DecisionTreeClassifier(), RandomForestClassifier(), XGBClassifier()]
params = [
    {
        'penalty': ['l1', 'l2'],
        'C': np.logspace(-4, 4, 10),
        'solver': ['liblinear']
    },
    {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
    },
    {
        'max_depth': np.linspace(1, 32, 32, endpoint=True),
        'min_samples_split': np.linspace(0.1, 1.0, 10, endpoint=True),
        'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True)
    },
    {
        'criterion': ['gini', 'entropy'],
        'min_samples_leaf': [1, 5, 10, 25],
        'min_samples_split': [10, 12, 16, 18],
        'n_estimators': [100, 700, 1500]
    },
    {
        'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.7]
    }
]


@click.command()
@click.argument('input_data', type=click.Path(exists=True))
@click.argument('output_model', type=click.Path())
def main(input_data, output_model):
    """ Runs modeling scripts using processed data (../raw) to
        create model. Model is saved as pickle (saved in ../models).
    """
    logger = logging.getLogger(__name__)
    logger.info('training model')

    data = DataSet(train_dir=input_data)
    train = data.get_train_set()
    X_train = data.get_features(train)
    y = data.get_label(train)

    clf = models[4]
    param_grid = params[4]

    model = Model.tune(clf, X_train, y, param_grid)
    model.save(output_model+model.name)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

