"""Save submissions for the kaggle challenge"""

import pandas as pd


def save_submission(submission: list[tuple[int, str]], output_path: str):
    """Save the submission to a csv file. This function takes as an input the direct output of the
    `summary` function"""
    submission_df = pd.DataFrame(submission, columns=["ID", "titles"])
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved at {output_path}")
