# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.data import DataSet
from src.data import DataWrangling


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    raw_data = DataSet(train_dir=input_filepath+'/train.csv', test_dir=input_filepath+'/test.csv')
    cleaning = DataWrangling(train_dir=output_filepath+'/train_clean.csv', test_dir=output_filepath+'/test_clean.csv')

    df_train = raw_data.get_train_set()
    df_test = raw_data.get_test_set()
    df_train_clean = cleaning.apply_preprocessing(df_train, target='Survived')
    df_test_clean = cleaning.apply_preprocessing(df_test, target='Survived')
    cleaning.processed_train_data(df_train_clean)
    cleaning.processed_test_data(df_test_clean)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
