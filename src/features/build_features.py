# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.data import DataSet
from src.features import TitanicPreProcessing


@click.command()
@click.argument('input_train', type=click.Path(exists=True))
@click.argument('input_test', type=click.Path(exists=True))
@click.argument('output_train', type=click.Path())
@click.argument('output_test', type=click.Path())
def main(input_train, input_test, output_train, output_test):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    raw_data = DataSet(input_train, input_test)

    df_train = raw_data.get_train_set()
    df_test = raw_data.get_test_set()

    TitanicPreProcessing(df_train, output_train)
    TitanicPreProcessing(df_test, output_test)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
