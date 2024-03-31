from src.load_data import load_data
from src.evaluation import summary
from src.submission import save_submission
from src.load_models import t5_base_fr_sum_cnndm, device


_, _, submission_df = load_data()

# Load the best model
DEVICE = device()
model, tokenizer, *_ = t5_base_fr_sum_cnndm()
model = model.to(DEVICE)

# Generate the summaries
output = summary(submission_df["text"], tokenizer, model, batch_size=4)

# Save the summaries
save_submission(output, "outputs/submissions/t5_base_fr_sum_cnndm_untrained.csv")
