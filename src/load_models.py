"""Load the different HuggingFace models"""

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)


def device():
    from torch.cuda import is_available

    return "cuda" if is_available() else "cpu"


def simple_load(model_name: str):
    return AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    ), AutoTokenizer.from_pretrained(model_name)


def mT5_multilingual_XLSum():
    """Example use:

    >>> output_ids = model.generate(
    >>>     input_ids=input_ids.to(device), max_length=200, no_repeat_ngram_size=2, num_beams=4
    >>> )[0]
    >>> summary = tokenizer.decode(
    >>>     output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    >>> )
    """
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        max_length=512,
        padding="max_length",
    )

    return model, tokenizer, data_collator


def mbart_mlsum_automatic_summarization():
    return simple_load("lincoln/mbart-mlsum-automatic-summarization")


def barthez_orangesum_title():
    return simple_load("moussaKam/barthez-orangesum-title")


def t5_base_fr_sum_cnndm():
    return simple_load("plguillou/t5-base-fr-sum-cnndm")


def flan_t5_large_dialogsum_fr():
    return simple_load("bofenghuang/flan-t5-large-dialogsum-fr")


# TODO
# https://huggingface.co/mrm8488/camembert2camembert_shared-finetuned-french-summarization
# https://huggingface.co/csebuetnlp/mT5_m2m_crossSum
# https://huggingface.co/hhhhzy/deltalm-base-xlsum
