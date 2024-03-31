"""Generate Labels for classification of articles"""

from transformers import AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
from src.metrics import Tokenizer


def labels_classification(
    text: pd.Series,
    tokenizer: Tokenizer,
    model: AutoModelForSequenceClassification,
    batch_size: int = 8,
):
    """Generate Labels using the FlauBERT model

    Using the standard representation defined in baseline part of notebook

    ## Input :
    - text : pd.Series : The text data for which the summaries are to be generated
        (the text column from the dataset)
    - t5_tokenizer : PreTrainedTokenizer : The tokenizer object
    - model : AutoModelForSeq2SeqLM : The model object

    ## Output :
    """
    summaries = []

    labels = model.config.id2label

    model.eval()

    for i in tqdm(range(0, len(text), batch_size)):
        batch = text[i : i + batch_size]
        input_text = tokenizer(
            batch.tolist(), return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(
            model.device  # type: ignore
        )

        logits = model(input_text).logits
        output = logits.argmax(dim=1)
        for idx, out in enumerate(output):
            summaries.append((i + idx, labels[out.item()]))

    return summaries
