from rouge_score import rouge_scorer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import pandas as pd

Tokenizer = PreTrainedTokenizerFast | PreTrainedTokenizer

SCORER = rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def gen_compute_metrics(tokenizer: Tokenizer):
    """Build a metric computing function for model fine-tuning"""
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def compute_metrics(eval_pred):  # type: ignore
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge_scores = [
            rouge.score(pred, label)["rougeL"].fmeasure
            for pred, label in zip(decoded_preds, decoded_labels)
        ]

        return {"rougeL_fmeasure": sum(rouge_scores) / len(rouge_scores)}

    return compute_metrics


def average_rouge_score(
    summaries: list[tuple[int, str]],
    titles: pd.Series,
    scorer: rouge_scorer.RougeScorer = SCORER,
):
    """Calculate the average rouge score for the summaries generated

    ## Input :
    - summaries : [(int, str)...] : A list of Tuples containing the index of
        the text and the summary generated
    - text : pd.Series : The text data for which the summaries are to be generated
        (the text column from the dataset)
    - scorer : rouge_scorer.RougeScorer : The Rouge Scorer object

    ## Output :
    - average_rouge : float : The average rouge score for the summaries generated
    """
    rouge_scores = []
    for idx, summary in summaries:
        reference = titles.iloc[idx]  # type: ignore
        rouge_scores.append(scorer.score(summary, reference)["rougeL"][2])
    return sum(rouge_scores) / len(rouge_scores)
