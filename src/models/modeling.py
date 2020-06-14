# Importing scikit-learn libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import numpy as np
import pandas as pd

from models import Model
from src.data import DataSet


class Modeling:
    def __init__(self, train, test, target):
        self.test = test
        self.X_train = self._get_features(train, target)
        self.y = self._get_label(train, target)
        self.X_test = self._get_features(test)
        self.clf_params = []
        self._init_clf_params()
        self.model_names = []
        self._modeling()

    @staticmethod
    def _get_features(data, target=None):
        if target:
            features = data.columns.drop([target, 'PassengerId']).values
        else:
            features = data.columns.drop(['PassengerId']).values
        return data[features]

    @staticmethod
    def _get_label(data, target):
        return data[target]

    def _init_clf_params(self):
        clf = LogisticRegression()
        params = {
            'penalty': ['l1', 'l2'],
            'C': np.logspace(-4, 4, 10),
            'solver': ['liblinear']
        }
        self.clf_params.append((clf, params))

        clf = SVC()
        params = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
        }
        self.clf_params.append((clf, params))

        clf = RandomForestClassifier()
        params = {
           'criterion': ['gini', 'entropy'],
           'min_samples_leaf': [1, 5, 10, 25],
           'min_samples_split': [10, 12, 16, 18],
           'n_estimators': [100, 700, 1500]
        }
        self.clf_params.append((clf, params))

    def _modeling(self):
        for clf, params in self.clf_params:
            model = Model.tune(clf, self.X_train, self.y, params)
            y_pred = model.predict(self.X_test)
            self.output(y_pred, model.name)
            model.save('/home/matteo@COPPET/Documents/data_science/Kaggle/titanic/models/'+model.name)
            self.model_names.append(model.name)

    def get_scores(self, scoring):
        scores = np.zeros((len(self.clf_params), 2))

        for i, model_name in enumerate(self.model_names):
            print(model_name)
            model = Model.load('/home/matteo@COPPET/Documents/'
                               'data_science/Kaggle/titanic/models/'+model_name)
            score = model.score(self.X_train, self.y, scoring=scoring)
            scores[i][0] = score.mean()
            scores[i][1] = score.std()

        scores = pd.DataFrame({'Model': self.model_names, 'Mean ' + scoring: scores[:, 0],
                               'Std of ' + scoring: scores[:, 1]})
        print('\n')
        print('Model scores:')
        print(scores)

    def output(self, predictions, name):
        output = pd.DataFrame({'PassengerId': self.test['PassengerId'], 'Survived': predictions})
        output.to_csv('/home/matteo@COPPET/Documents/data_science/Kaggle/titanic'
                      '/models/submission_{}.csv'.format(name), index=False)


clean_data = DataSet(train_dir='/home/matteo@COPPET/Documents/data_science/'
                               'Kaggle/titanic/data/processed/train_clean.csv',
                     test_dir='/home/matteo@COPPET/Documents/data_science/'
                               'Kaggle/titanic/data/processed/test_clean.csv')
X_train = clean_data.get_train_set()
X_test = clean_data.get_test_set()

modeling = Modeling(X_train, X_test, 'Survived')
modeling.get_scores('accuracy')
