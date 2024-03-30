# # Baseline For Testing

from operator import itemgetter
import pandas as pd
from rouge_score import rouge_scorer

from typing import List, Tuple, Dict

import torch
from tqdm import tqdm

from datasets.dataset_dict import DatasetDict
from datasets import Dataset


from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# PATH VARIABLES

TRAIN_PATH = "./data/train.csv"
VALIDATION_PATH = "./data/validation.csv"
TEST_PATH = "./data/test_text.csv"

# BASE MODEL
BASE_MODEL = "t5-small"
# AVAILABLE_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]
CHECKPOINT_NUMBER = 2676


def load_data(
    train_path: str, validation_path: str, test_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


# Function that generates summaries using LEAD-N
def lead_summary(text: pd.Series) -> List[Tuple[int, str]]:
    """Generate summaries using the LEAD-N method

    ## Input :
    - text : pd.Series : The text data for which the summaries are to be generated (the text column from the dataset)

    ## Output :
    - summaries : List[Tuple[int, str]] : A list of Tuples containing the index of the text and the summary generated using the LEAD-N method
    """
    summaries = []
    for idx, row in text.items():
        sentences = row.split(".")
        summaries.append((idx, sentences[0] + "."))
    return summaries


# Function that generates summaries using EXT-ORACLE
def ext_oracle_summary(
    text: pd.Series,
    titles: pd.Series,
    scorer: rouge_scorer.RougeScorer,
) -> List[Tuple[int, str]]:
    """Generate summaries using the EXT-ORACLE method

    ## Input :
    - text : pd.Series : The text data for which the summaries are to be generated (the text column from the dataset)
    - titles : pd.Series : The titles of the text data (the titles column from the dataset)
    - scorer : rouge_scorer.RougeScorer : The Rouge Scorer object

    ## Output :
    - summaries : List[Tuple[int, str]] : A list of Tuples containing the index of the text and the summary generated using the EXT-ORACLE method
    """
    summaries = []
    for idx, row in text.items():
        sentences = row.split(".")
        reference = titles.iloc[idx]  # type: ignore
        rs = [scorer.score(sentence, reference)["rougeL"][2] for sentence in sentences]
        index, _ = max(enumerate(rs), key=itemgetter(1))
        summaries.append((idx, sentences[index]))
    return summaries


def average_rouge_score(
    summaries: List[Tuple[int, str]],
    titles: pd.Series,
    scorer: rouge_scorer.RougeScorer,
):
    """Calculate the average rouge score for the summaries generated

    ## Input :
    - summaries : [(int, str)...] : A list of Tuples containing the index of the text and the summary generated
    - text : pd.Series : The text data for which the summaries are to be generated (the text column from the dataset)
    - scorer : rouge_scorer.RougeScorer : The Rouge Scorer object

    ## Output :
    - average_rouge : float : The average rouge score for the summaries generated
    """
    rouge_scores = []
    for idx, summary in summaries:
        reference = titles.iloc[idx]  # type: ignore
        rouge_scores.append(scorer.score(summary, reference)["rougeL"][2])
    return sum(rouge_scores) / len(rouge_scores)


def preprocess_text(
    text: str,
    title: str,
    t5_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    prefix: str = "summarize: ",
    max_input_length: int = 1024,
    max_target_length: int = 64,
) -> Dict[str, torch.Tensor]:
    """Preprocess the text data

    ## Input :
    - text : str : The text data to be preprocessed
    - title : str : The title of the text data, The Target
    - prefix : str : The prefix to be added to the text data as T5 model can be used for translation as well
    - max_input_length : int : The maximum length of the input text
    - max_target_length : int : The maximum length of the target text
    - tokenizer : t5_tokenizer : The tokenizer object

    ## Output :
    - model_inputs : Dict[str, Union[torch.Tensor, None]] : The model inputs
    """
    inputs = t5_tokenizer(
        f"{prefix} {text}",
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
    )

    targets = t5_tokenizer(
        title,
        max_length=max_target_length,
        padding="max_length",
        truncation=True,
    )

    model_inputs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": targets.input_ids,
    }

    return model_inputs


def preprocess_from_df(
    df: pd.DataFrame, tokenizer_model: PreTrainedTokenizerFast | PreTrainedTokenizer
):
    dataframe_list = []
    for i in range(len(df)):
        dataframe_list.append(
            preprocess_text(df["text"].iloc[i], df["titles"].iloc[i], tokenizer_model)
        )

    return Dataset.from_list(dataframe_list)


# Construct metric
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def compute_metrics(eval_pred, tokenizer_model):
    predictions, labels = eval_pred
    decoded_preds = tokenizer_model.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer_model.batch_decode(labels, skip_special_tokens=True)

    rouge_scores = [
        rouge.score(pred, label)["rougeL"].fmeasure
        for pred, label in zip(decoded_preds, decoded_labels)
    ]

    return {"rougeL_fmeasure": sum(rouge_scores) / len(rouge_scores)}


def metric_function_generator(tokenize_model):
    return lambda eval_pred: compute_metrics(eval_pred, tokenize_model)


# ### Running


def train_model_from_checkpoint(model_checkpoint: str):
    """
    Pick a checkpoint from the list available on HunggingFace and finetune it on the dataset
    """

    # Load the model and Tokenizer
    # This shoud load the T5TokenizerFast from the transformers library
    t5_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    print(f"Tokenizer Loaded : {model_checkpoint}")
    # t5_tokenizer = T5TokenizerFast.from_pretrained(model_checkpoint)
    # otherwise we can use thisone and compare the results
    # t5_tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    # This should load the T5ForConditionalGeneration model from the transformers library
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    print(f"Model Loaded : {model_checkpoint}")

    batch_size = 8
    model_name = model_checkpoint.split("/")[-1]
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./outputs/{model_name}-finetuned",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        num_train_epochs=5,
        eval_steps=2,
        save_steps=2,
        warmup_steps=1,
        # overwrite_output_dir=True,
        save_total_limit=2,
    )

    # Construct the DataSet
    #
    print("Building Dataset...", end="\r")
    data_collator = DataCollatorForSeq2Seq(t5_tokenizer, model=model)
    train_dataset = preprocess_from_df(train_df, t5_tokenizer)
    validation_dataset = preprocess_from_df(validation_df, t5_tokenizer)
    total_dataset = DatasetDict(
        {"train": train_dataset, "validation": validation_dataset}
    )
    print("Building Dataset... Done!")

    # Generate the metric function with the right tokenizer
    metric_function = metric_function_generator(t5_tokenizer)

    # Construct the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=metric_function,
        train_dataset=total_dataset["train"],
        eval_dataset=total_dataset["validation"],
    )

    # Train the model
    print("Training Model...")
    trainer.train()

    return model, t5_tokenizer


# ### Evaluation


def t5_summary(
    text: pd.Series, t5_tokenizer: PreTrainedTokenizer, model: AutoModelForSeq2SeqLM
):
    """Generate summaries using the T5 model

    Using the standard representation defined in baseline part of notebook

    ## Input :
    - text : pd.Series : The text data for which the summaries are to be generated (the text column from the dataset)
    - t5_tokenizer : PreTrainedTokenizer : The tokenizer object
    - model : AutoModelForSeq2SeqLM : The model object

    ## Output :
    - summaries : List[Tuple[int, str]] : A list of Tuples containing the index of the text and the summary generated using the T5 model
    """
    summaries = []
    for idx, row in tqdm(text.items()):
        input_text = t5_tokenizer(row, return_tensors="pt").input_ids.to(model.device)
        output = model.generate(
            input_text,
            max_length=64,
            early_stopping=True,
            num_return_sequences=1,
        )
        summaries.append(
            (idx, t5_tokenizer.decode(output[0], skip_special_tokens=True))
        )
    return summaries


def inference_once(text: str, model, t5_tokenizer):
    """Use a model to generate a title for a single text input"""
    input_text = t5_tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(
        input_text,
        max_length=64,
        early_stopping=True,
        num_return_sequences=1,
    )
    return t5_tokenizer.decode(output[0], skip_special_tokens=True)


def submission_from_df(model, t5_tokenizer, test_df):
    """Create a submission file from a DataFrame of text data ready for kaggle"""
    t5_summary_kaggle = t5_summary(test_df["text"], t5_tokenizer, model)
    t5_summary_kaggle_df = pd.DataFrame(t5_summary_kaggle, columns=["ID", "titles"])
    t5_summary_kaggle_df.to_csv(
        "./outputs/submissions//t5_summary_kaggle.csv", index=False
    )


def submission_from_pretrained(
    model_checkpoint: str,
    checkpoint_number: int,
    test_df: pd.DataFrame,
    output_path: str | None = None,
):
    """Create a submission file from a DataFrame of text data ready for kaggle"""

    # Load the model and Tokenizer
    t5_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        f"./outputs/{model_checkpoint}-finetuned/checkpoint-{checkpoint_number}"
    )

    # Generate the submission
    t5_summary_kaggle = t5_summary(test_df["text"], t5_tokenizer, model)  # type: ignore
    t5_summary_kaggle_df = pd.DataFrame(t5_summary_kaggle, columns=["ID", "titles"])

    # Save the submission
    #
    if output_path is None:
        output_path = f"./outputs/submissions/{model_checkpoint}-finetuned-{checkpoint_number}.csv"

    t5_summary_kaggle_df.to_csv(output_path, index=False)
    print(f"Submission saved at {output_path}")


def load_model(model_checkpoint: str, checkpoint_number: int):
    """Load a model from a checkpoint"""
    t5_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        f"./outputs/{model_checkpoint}-finetuned/checkpoint-{checkpoint_number}"
    )
    return model, t5_tokenizer


###

if __name__ == "__main__":
    # Load the data
    train_df, validation_df, test_df = load_data(TRAIN_PATH, VALIDATION_PATH, TEST_PATH)

    # Train the model
    # model, t5_tokenizer = train_model_from_checkpoint("t5-small")
    #
    # Load the model
    model, t5_tokenizer = load_model(BASE_MODEL, CHECKPOINT_NUMBER)

    # Generate the T5 summaries
    t5_summaries = t5_summary(validation_df["text"], t5_tokenizer, model)  # type: ignore
    # Calculate the average rouge score for the T5 summaries
    t5_rouge_score = average_rouge_score(t5_summaries, validation_df["titles"], rouge)  # type: ignore
    print(f"T5 Rouge Score : {t5_rouge_score}")

    # Generate the submission
    submission_from_df(model, t5_tokenizer, test_df)
    submission_from_pretrained("t5-small", CHECKPOINT_NUMBER, test_df)
