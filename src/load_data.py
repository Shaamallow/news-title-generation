"""Load the data from the datasets"""

import pandas as pd

TRAIN_PATH = "./data/train.csv"
VALIDATION_PATH = "./data/validation.csv"
TEST_PATH = "./data/test_text.csv"


def load_data(
    train_path: str = TRAIN_PATH,
    validation_path: str = VALIDATION_PATH,
    test_path: str = TEST_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the data from the files
    ## Input :
    - train_path : str : The path to the training data
    - validation_path : str : The path to the validation data
    - test_path : str : The path to the test data
    ## Output :
    - train_df : pd.DataFrame : The training data
    - validation_df : pd.DataFrame : The validation data
    - test_df : pd.DataFrame : The test data
    """

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    return train_df, validation_df, test_df
