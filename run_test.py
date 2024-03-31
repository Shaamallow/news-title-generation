# Test my things

from src.load_models import mT5_multilingual_XLSum
from src.tester import do_test

model, tokenizer, *_ = mT5_multilingual_XLSum()

do_test(model, tokenizer, batch_size=1)
