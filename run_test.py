# Test my things

from src.load_models import mbart_mlsum_automatic_summarization
from src.tester import do_test

model, tokenizer, *_ = mbart_mlsum_automatic_summarization()

do_test(model, tokenizer, batch_size=1)
