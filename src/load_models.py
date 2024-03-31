"""Load the different HuggingFace models.
We include their raw scores on the validation dataset before training.
"""

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    RobertaTokenizerFast,
    EncoderDecoderModel,
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
    # Average Rouge Score: 0.20527144694133828

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
    # Average Rouge Score: 0.20474941262252572
    return simple_load("lincoln/mbart-mlsum-automatic-summarization")


def barthez_orangesum_title():
    # Average Rouge Score: 0.17222983224367533
    return simple_load("moussaKam/barthez-orangesum-title")


def t5_base_fr_sum_cnndm():
    # Average Rouge Score: 0.20781614162134016
    return simple_load("plguillou/t5-base-fr-sum-cnndm")


def flan_t5_large_dialogsum_fr():
    # Average Rouge Score: 0.15820893720532173
    # TOO LARGE
    return simple_load("bofenghuang/flan-t5-large-dialogsum-fr")


def camember2camember():
    """
    >>> def generate_summary(text):
    >>>     inputs = tokenizer([text], padding="max_length",
    >>>                        truncation=True, max_length=512, return_tensors="pt")
    >>>     input_ids = inputs.input_ids.to(device)
    >>>     attention_mask = inputs.attention_mask.to(device)
    >>>     output = model.generate(input_ids, attention_mask=attention_mask)
    >>>     return tokenizer.decode(output[0], skip_special_tokens=True)
    """
    # Average Rouge Score: 0.20582048179444393
    model_name = "mrm8488/camembert2camembert_shared-finetuned-french-summarization"
    model = EncoderDecoderModel.from_pretrained(model_name)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

    return model, tokenizer


def mT5_m2m_crossSum():
    # Average Rouge Score: 0.20217067318519055
    model_name = "csebuetnlp/mT5_m2m_crossSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return model, tokenizer
