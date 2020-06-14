# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from src.data import DataSet
from src.models import Model


@click.command()
@click.argument('input_data', type=click.Path(exists=True))
@click.argument('input_model', type=click.Path(exists=True))
@click.argument('output_prediction', type=click.Path())
def main(input_data, input_model, output_prediction):
    """ Runs modeling scripts using model pickle (../models) to predict
        outcomes. Outcomes file is saved as .csv (saved in ../models).
    """
    logger = logging.getLogger(__name__)
    logger.info('predicting outcomes')

    data = DataSet(test_dir=input_data)
    test = data.get_test_set()
    X_test = data.get_features(test)

    model = Model.load(input_model+'XGBClassifier')
    y_pred = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_pred})
    output.to_csv(output_prediction+'submission_{}.csv'.format(model.name), index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

