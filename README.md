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
├── data
│   ├── test_text.csv
│   ├── train.csv
│   └── validation.csv
├── docs
│   ├── dataset_polar_visualization.png
│   ├── labels_repartition.png
│   ├── pca_labels_flaubert.png
│   ├── pca_labels.png
│   ├── pca_rouge.png
│   ├── polar_rouge_score.png
│   └── rouge_distribution.png
├── IN582_2024_Challenge.pdf
├── notebooks
│   └── baseline.ipynb
├── outputs
│   ├── bart-base-finetuned-3 # Best BART fine-tuned model
│   ├── submissions 
│   ├── t5-small-finetuned # 1st T5 fine-tuned model
│   └── t5-small-finetuned-2 # Best T5 fine-tuned model
├── pyproject.toml
├── README.md
├── requirements.txt
├── run_dataset.py # Script to generate the dataset visualization plots from the report
├── run_fine_tuning.py # Script to fine-tune the models
├── run_submission.py # Script to generate the submissions
├── run_test.py 
├── src
│   ├── evaluation.py
│   ├── labels.py
│   ├── load_data.py
│   ├── load_models.py
│   ├── metrics.py
│   ├── __pycache__
│   ├── submission.py
│   └── tester.py
├── summarization_baselines.py
└── t5_pipeline.py # Old script to fine-tune the T5 model
```
