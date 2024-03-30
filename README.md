# INF 582 Data Challenge

Summarize news articles by giving them relevant titles.

## Quickstart

This project was made using Python 3.11.
All necessary dependencies are listed in the `requirements.txt` file. To install them, run:

```bash
pip install -r requirements.txt
```

It might be necessary to install `torch[transformers]` separately. To do so, run:
```bash
pip install "torch[transformers]"
```
Edit the requirements in case something is missing

## Project Structure

```bash
.
├── data # Source dataset
│   ├── test_text.csv
│   ├── train.csv
│   └── validation.csv
├── IN582_2024_Challenge.pdf
├── notebooks
│   └── baseline.ipynb  # Jupyter Notebook with pipeline for t5 model training and evaluation
├── outputs
│   ├── submissions
│   │   ├── ext_oracle_submission.csv
│   │   ├── lead_submission.csv
│   │   └── t5_summary_kaggle.csv
│   └── t5-small-finetuned # Fine-tuned model checkpoints
│       └── checkpoint-2676 # Current best model
│           ├── config.json
│           ├── generation_config.json
│           ├── model.safetensors
│           ├── optimizer.pt
│           ├── rng_state.pth
│           ├── scheduler.pt
│           ├── trainer_state.json
│           └── training_args.bin
├── pyproject.toml
├── README.md
├── requirements.txt
├── summarization_baselines.py
└── t5_pipeline.py # Main pipeline for training and evaluation 
```
