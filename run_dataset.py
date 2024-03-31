# Run Analysis on the dataset

import pandas as pd
import numpy as np

from tqdm import tqdm

from src.load_data import load_data
from src.load_models import device, t5_base_fr_sum_cnndm
from src.labels import labels_classification

from src.evaluation import embeddings, summary

from rouge_score import rouge_scorer

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def generate_labels():
    """Generate classification labels for the dataset using Flaubert topic classification model"""
    train_df, validation_df, test_df = load_data()

    # Now compute the genre of each text using a hugging face model
    tokenizer = AutoTokenizer.from_pretrained(
        "lincoln/flaubert-mlsum-topic-classification"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "lincoln/flaubert-mlsum-topic-classification"
    )

    DEVICE = device()
    model = model.to(DEVICE)

    labels_train = labels_classification(train_df["text"], tokenizer, model, 4)  # type: ignore
    validation_labels = labels_classification(
        validation_df["text"],  # type: ignore
        tokenizer,
        model,
        4,
    )
    test_labels = labels_classification(test_df["text"], tokenizer, model, 4)  # type: ignore

    # add a colmumn to the dataframe
    for i, label in labels_train:
        train_df.at[i, "genre"] = label

    for i, label in validation_labels:
        validation_df.at[i, "genre"] = label

    for i, label in test_labels:
        test_df.at[i, "genre"] = label

    # Save the dataframes
    train_df.to_csv("data/train_labels.csv", index=False)
    validation_df.to_csv("data/validation_labels.csv", index=False)
    test_df.to_csv("data/test_labels.csv", index=False)


def labels_analysis():
    """Generate a pie plot of the labels distribution"""

    TRAIN_PATH = "data/train_labels.csv"
    VALIDATION_PATH = "data/validation_labels.csv"
    TEST_PATH = "data/test_labels.csv"
    train_df, validation_df, test_df = load_data(TRAIN_PATH, VALIDATION_PATH, TEST_PATH)

    _, axs = plt.subplots(1, 3, figsize=(20, 5))
    plt.rcParams["font.family"] = "cmr10"
    plt.rc("axes", unicode_minus=False)

    # pie plot
    train_df["genre"].value_counts().plot.pie(
        ax=axs[0], autopct="%1.1f%%", cmap="viridis"
    )
    axs[0].set_title("Train Data", fontproperties="cmr10", fontsize=20)
    axs[0].set_ylabel("Count : " + str(len(train_df)), fontproperties="cmr10")

    validation_df["genre"].value_counts().plot.pie(
        ax=axs[1], autopct="%1.1f%%", cmap="viridis"
    )
    axs[1].set_title("Validation Data", fontproperties="cmr10", fontsize=20)
    axs[1].set_ylabel("Count : " + str(len(validation_df)), fontproperties="cmr10")

    test_df["genre"].value_counts().plot.pie(
        ax=axs[2], autopct="%1.1f%%", cmap="viridis"
    )
    axs[2].set_title("Test Data", fontproperties="cmr10", fontsize=20)
    axs[2].set_ylabel("Count : " + str(len(test_df)), fontproperties="cmr10")

    # Save
    plt.savefig("docs/labels_repartition.png", bbox_inches="tight", dpi=300)


def add_embeddings():
    """Add embeddings given a tokenizer to the dataframes"""
    TRAIN_PATH = "data/train_labels.csv"
    VALIDATION_PATH = "data/validation_labels.csv"
    TEST_PATH = "data/test_labels.csv"

    train_df, validation_df, test_df = load_data(TRAIN_PATH, VALIDATION_PATH, TEST_PATH)
    # _, tokenizer = t5_base_fr_sum_cnndm()
    tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_cased")

    # generate the embeddings with the tokenizer
    train_df["embeddings"] = embeddings(train_df.text, tokenizer, 16)
    validation_df["embeddings"] = embeddings(validation_df.text, tokenizer, 16)
    test_df["embeddings"] = embeddings(test_df.text, tokenizer, 16)

    # save the dataframes
    train_df.to_csv("data/train_df_embeddings_flaubert.csv", index=False)
    validation_df.to_csv("data/validation_df_embeddings_flaubert.csv", index=False)
    test_df.to_csv("data/test_df_embeddings_flaubert.csv", index=False)


def add_generated_title():
    """Add generated title to the dataframes"""
    TRAIN_PATH = "data/train_df_embeddings.csv"
    VALIDATION_PATH = "data/validation_df_embeddings.csv"
    TEST_PATH = "data/test_df_embeddings.csv"
    _, validation_df, _ = load_data(TRAIN_PATH, VALIDATION_PATH, TEST_PATH)
    model, tokenizer = t5_base_fr_sum_cnndm()

    DEVICE = device()
    model = model.to(DEVICE)

    validation_summaries = summary(validation_df.text, tokenizer, model, batch_size=64)

    for i, title in tqdm(validation_summaries):
        validation_df.at[i, "generated_title"] = title

    print(validation_df.head())
    validation_df.to_csv("data/validation_df_generated.csv", index=False)


def generate_visualization():
    """Generate visualization of both labels and size of the text from the dataset"""
    TRAIN_PATH = "data/train_labels.csv"
    VALIDATION_PATH = "data/validation_labels.csv"
    TEST_PATH = "data/test_labels.csv"

    # no need for the embeddings this time
    # each row is a point in the 2D space
    # with an orientation defined from the labels (classes from the flaubert model)
    # the distance from the center of the plan is defined by the size of the text
    # add a bit of noise to the angles to avoid overlapping
    # and the color will be the labels
    train_df, validation_df, test_df = load_data(TRAIN_PATH, VALIDATION_PATH, TEST_PATH)

    total_df = pd.concat([train_df, validation_df, test_df])

    # Compute the size of the text
    total_df["size"] = total_df["text"].apply(lambda x: len(x.split()))

    # Compute the angle of the text
    # add a bit of noise to avoid overlapping
    classes = total_df["genre"].unique()
    n_classes = len(classes)

    N_JITTER = 0.23

    total_df["angle"] = total_df["genre"].apply(
        lambda x: 2 * 3.1415 * classes.tolist().index(x) / n_classes
        + np.random.uniform(-N_JITTER, N_JITTER)
    )

    # Compute the distance from the center
    total_df["distance"] = total_df["size"]

    # remove the outliers
    # total_df = total_df[total_df["distance"] < 2000]

    # Now scatter plot
    total_df["color"] = total_df["genre"].apply(lambda x: classes.tolist().index(x))

    plt.rcParams["font.family"] = "cmr10"
    plt.rc("axes", unicode_minus=False)

    _ = plt.figure(figsize=(20, 10))
    ax = plt.subplot(111, polar=True)

    MAX_SIZE = 5000
    ax.set_rlim(10, MAX_SIZE)  # type: ignore
    ax.set_rscale("symlog")  # type: ignore

    x_ticks = np.linspace(0, 2 * 3.1415, n_classes, endpoint=False)
    y_ticks = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    y_ticks = [tick * 10**i for i in range(3) for tick in y_ticks]

    y_ticks = [tick for tick in y_ticks if tick < MAX_SIZE]

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.set_xticklabels(classes, fontproperties="cmr10", fontsize=15)

    ax.grid(True, which="major", ls="--", color="black", alpha=0.5)
    ax.set_axisbelow(True)

    ax.scatter(
        total_df["angle"],
        total_df["distance"],
        c=total_df["color"],
        cmap="viridis",
        alpha=0.75,
    )

    plt.title("Visualization of the dataset", fontproperties="cmr10", fontsize=20)
    plt.savefig("docs/dataset_polar_visualization.png", bbox_inches="tight", dpi=300)
    # plt.show()

    # print(total_df.head())


def generate_visualization_rouge():
    """Generate Visualization for the ROUGE score, size and labels from the dataset"""
    # no need for the embeddings this time
    # each row is a point in the 2D space
    # with an orientation defined from the labels (classes from the flaubert model)
    # the distance from the center of the plan is defined by the size of the text
    # add a bit of noise to the angles to avoid overlapping
    # and the color will be the labels
    TRAIN_PATH = "data/train_df_embeddings.csv"
    VALIDATION_PATH = "data/validation_df_generated.csv"
    TEST_PATH = "data/test_df_embeddings.csv"

    # Perfom only on the validation set for proper ROUGE score (avoid overfitting on train data)
    _, validation_df, _ = load_data(TRAIN_PATH, VALIDATION_PATH, TEST_PATH)

    # Compute the rouge score for eaach title
    SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = []
    # We will only use the fmeasure of the scorer as we are graded on the fscore avg
    for i, row in validation_df.iterrows():
        rouge_scores.append(
            SCORER.score(row.generated_title, row.titles)["rougeL"].fmeasure
        )

    validation_df["rouge_score"] = rouge_scores
    total_df = validation_df

    # Compute the size of the text
    total_df["size"] = total_df["text"].apply(lambda x: len(x.split()))

    # Compute the angle of the text
    # add a bit of noise to avoid overlapping
    classes = total_df["genre"].unique()
    n_classes = len(classes)

    N_JITTER = 0.23

    total_df["angle"] = total_df["genre"].apply(
        lambda x: 2 * 3.1415 * classes.tolist().index(x) / n_classes
        + np.random.uniform(-N_JITTER, N_JITTER)
    )

    # Compute the distance from the center
    total_df["distance"] = total_df["size"]

    # Now scatter plot
    total_df["color"] = total_df["rouge_score"]

    plt.rcParams["font.family"] = "cmr10"
    plt.rc("axes", unicode_minus=False)

    _ = plt.figure(figsize=(20, 10))
    ax = plt.subplot(111, polar=True)

    MAX_SIZE = 5000
    ax.set_rlim(10, MAX_SIZE)  # type: ignore
    ax.set_rscale("symlog")  # type: ignore

    x_ticks = np.linspace(0, 2 * 3.1415, n_classes, endpoint=False)
    y_ticks = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    y_ticks = [tick * 10**i for i in range(3) for tick in y_ticks]

    y_ticks = [tick for tick in y_ticks if tick < MAX_SIZE]

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.set_xticklabels(classes, fontproperties="cmr10", fontsize=15)

    ax.grid(True, which="major", ls="--", color="black", alpha=0.5)
    ax.set_axisbelow(True)

    scatter = ax.scatter(
        total_df["angle"],
        total_df["distance"],
        c=total_df["color"],
        cmap="Reds",
        alpha=0.75,
    )
    plt.colorbar(scatter, label="ROUGE score")
    plt.title("Visualization ROUGE score", fontproperties="cmr10", fontsize=20)
    plt.savefig("docs/polar_rouge_score.png", bbox_inches="tight", dpi=300)
    # plt.show()


def pca_analysis(save: bool = False):
    """Perform PCA analysis on the embeddings"""

    print("Performing PCA analysis on the embeddings")

    TRAIN_PATH = "data/train_df_embeddings.csv"
    VALIDATION_PATH = "data/validation_df_embeddings.csv"
    TEST_PATH = "data/test_df_embeddings.csv"

    print("Loading data")
    train_df, validation_df, test_df = load_data(TRAIN_PATH, VALIDATION_PATH, TEST_PATH)

    all_df = [train_df, validation_df, test_df]
    total_df = pd.concat(all_df)

    classes = total_df["genre"].unique()
    total_df["color"] = total_df["genre"].apply(lambda x: classes.tolist().index(x))

    pca = PCA(n_components=2)

    print("Fitting PCA")

    X = np.array(total_df["embeddings"].apply(lambda x: x[1:-1].split(",")))
    X = np.array([np.array(x) for x in X])
    X = X.astype(float)

    X_pca = pca.fit_transform(X)

    plt.rcParams["font.family"] = "cmr10"
    plt.rc("axes", unicode_minus=False)

    _ = plt.figure(figsize=(10, 10))

    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=total_df["color"], cmap="viridis")
    plt.xlabel("PCA 1", fontproperties="cmr10")
    plt.ylabel("PCA 2", fontproperties="cmr10")

    plt.title(
        "PCA analysis of the embeddings with color classes", fontproperties="cmr10"
    )
    handles = scatter.legend_elements()[0]
    plt.legend(handles, classes, title="Classes", prop={"family": "cmr10"})
    plt.savefig("docs/pca_labels_flaubert_2.png", bbox_inches="tight", dpi=300)
    # plt.show()

    if save:
        # Save PCA values
        total_df["pca1"] = X_pca[:, 0]
        total_df["pca2"] = X_pca[:, 1]

        total_df.to_csv("data/total_df_embeddings.csv", index=False)


def pca_analysis_rouge(save: bool = False):
    """Perform PCA analysis on the embeddings but color by rouge score"""
    TRAIN_PATH = "data/train_df_embeddings_flaubert.csv"
    VALIDATION_PATH = "data/validation_df_embeddings_flaubert.csv"
    TEST_PATH = "data/test_df_embeddings_flaubert.csv"

    # Perfom only on the validation set for proper ROUGE score (avoid overfitting on train data)
    _, validation_df, _ = load_data(TRAIN_PATH, VALIDATION_PATH, TEST_PATH)

    # Compute the rouge score for eaach title
    SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = []
    # We will only use the fmeasure of the scorer as we are graded on the fscore avg
    for i, row in validation_df.iterrows():
        rouge_scores.append(
            SCORER.score(row.generated_title, row.titles)["rougeL"].fmeasure
        )

    validation_df["rouge_score"] = rouge_scores

    if save:
        validation_df.to_csv("data/validation_df_rouge.csv", index=False)

    # Scale colors from BLUE (0) to RED (1) for the ROUGE score
    validation_df["color"] = validation_df["rouge_score"]

    pca = PCA(n_components=2)

    print("Fitting PCA")

    X = np.array(validation_df["embeddings"].apply(lambda x: x[1:-1].split(",")))
    X = np.array([np.array(x) for x in X])
    X = X.astype(float)

    X_pca = pca.fit_transform(X)

    plt.rcParams["font.family"] = "cmr10"
    plt.rc("axes", unicode_minus=False)

    _ = plt.figure(figsize=(20, 10))

    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=validation_df["color"], cmap="coolwarm"
    )
    plt.xlabel("PCA 1", fontproperties="cmr10")
    plt.ylabel("PCA 2", fontproperties="cmr10")

    plt.title(
        "PCA analysis of the embeddings with color ROUGE score",
        fontproperties="cmr10",
        fontsize=20,
    )
    plt.colorbar(scatter, label="ROUGE score")
    plt.savefig("docs/pca_rouge.png", bbox_inches="tight", dpi=300)
    # plt.show()
    plt.clf()

    # Now plot the ROUGE score distribution for each class
    classes = validation_df["genre"].unique()
    # span on 2 rows
    fig, axs = plt.subplots(3, len(classes) // 3, figsize=(20, 10))
    plt.rcParams["font.family"] = "cmr10"
    plt.rc("axes", unicode_minus=False)

    print(axs.shape)

    plt.subplots_adjust(hspace=0.5)
    for i, genre in enumerate(classes):
        genre_df = validation_df[validation_df["genre"] == genre]
        axs[i // 3, i % 3].hist(genre_df["rouge_score"], bins=20)
        axs[i // 3, i % 3].set_title(genre, fontproperties="cmr10", fontsize=15)
        axs[i // 3, i % 3].set_ylabel("Count")
        axs[i // 3, i % 3].set_xlabel("ROUGE score")
        axs[i // 3, i % 3].set_xlim(0, 1)

    plt.suptitle(
        "ROUGE score distribution for each class", fontproperties="cmr10", fontsize=20
    )
    plt.savefig("docs/rouge_distribution.png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    # generate_labels()
    # labels_analysis()
    # add_embeddings()
    # generate_visualization()
    # generate_visualization_rouge()
    # pca_analysis()
    # add_generated_title()
    # pca_analysis_rouge()
    pass
