"""Function to test a model on the whole dataset"""

from transformers import AutoModelForSeq2SeqLM

from src.load_data import load_data
from src.load_models import device
from src.evaluation import t5_summary
from src.metrics import average_rouge_score, Tokenizer


def do_test(model: AutoModelForSeq2SeqLM, tokenizer: Tokenizer, batch_size: int = 1):

    DEVICE = device()

    _, validation_df, _ = load_data()

    model = model.to(DEVICE)  # type: ignore

    summaries = t5_summary(
        validation_df["text"], tokenizer, model, batch_size=batch_size
    )

    average_rouge = average_rouge_score(summaries, validation_df["title"])

    print(f"Average Rouge Score: {average_rouge}")
