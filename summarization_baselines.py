# pip install rouge_score
from operator import itemgetter
import pandas as pd
from rouge_score import rouge_scorer

# Read the validation data
validation_df = pd.read_csv("data/validation.csv")

rouge_scores = []
# Initialize Rouge Scorer
scorer = rouge_scorer.RougeScorer(["rougeL"])


# Function that generates summaries using LEAD-N
def lead_summary(text: pd.Series, titles: pd.Series, scorer: rouge_scorer.RougeScorer):
    summaries = []
    for idx, row in text.items():
        sentences = row.split(".")
        summaries.append([idx, sentences[0] + "."])
    return summaries


# Function that generates summaries using EXT-ORACLE
def ext_oracle_summary(
    text: pd.Series,
    titles: pd.Series,
    scorer: rouge_scorer.RougeScorer,
):
    summaries = []
    for idx, row in text.items():
        sentences = row.split(".")
        reference = titles.iloc[idx]  # type: ignore
        rs = [scorer.score(sentence, reference)["rougeL"][2] for sentence in sentences]
        index, _ = max(enumerate(rs), key=itemgetter(1))
        summaries.append([idx, sentences[index]])
    return summaries


lead_summaries = lead_summary(validation_df["text"], validation_df["titles"], scorer)
ext_oracle_summaries = ext_oracle_summary(
    validation_df["text"], validation_df["titles"], scorer
)

lead_rouge = []
ext_oracle_rouge = []
# Calculate the rouge-l score for each of the generated summaries compared to the original titles
for idx, title in validation_df["titles"].items():
    lead_rouge.append(
        scorer.score(lead_summaries[idx][1], title)["rougeL"][2]  # type: ignore
    )
    ext_oracle_rouge.append(
        scorer.score(ext_oracle_summaries[idx][1], title)["rougeL"][2]  # type: ignore
    )

avg_rouge_score_lead = sum(lead_rouge) / len(lead_rouge)
avg_rouge_score_ext_oracle = sum(ext_oracle_rouge) / len(ext_oracle_rouge)

print("Average Rouge-L F-Score with LEAD-1: ", avg_rouge_score_lead)
print("Average Rouge-L F-Score with EXT-ORACLE:", avg_rouge_score_ext_oracle)

# Store the generated summaries in the Kaggle-accepted format
lead_submission_df = pd.DataFrame(lead_summaries, columns=["ID", "titles"])
ext_oracle_submission_df = pd.DataFrame(ext_oracle_summaries, columns=["ID", "titles"])
lead_submission_df.to_csv("lead_submission.csv", index=False)
ext_oracle_submission_df.to_csv("ext_oracle_submission.csv", index=False)
