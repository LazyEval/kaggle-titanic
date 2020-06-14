from src.features import TitanicPreProcessing


def test_impute():
    assert TitanicPreProcessing.impute('ok', 'cunt')