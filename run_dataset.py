# Run Analysis on the dataset

import pandas as pd
import numpy as np
from src.load_data import load_data
from src.load_models import device, t5_base_fr_sum_cnndm
from src.labels import labels_classification

from src.evaluation import embeddings

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification, trainer


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
    _, tokenizer = t5_base_fr_sum_cnndm()

    # generate the embeddings with the tokenizer
    train_df["embeddings"] = embeddings(train_df.text, tokenizer, 16)
    validation_df["embeddings"] = embeddings(validation_df.text, tokenizer, 16)
    test_df["embeddings"] = embeddings(test_df.text, tokenizer, 16)

    # save the dataframes
    train_df.to_csv("data/train_df_embeddings.csv", index=False)
    validation_df.to_csv("data/validation_df_embeddings.csv", index=False)
    test_df.to_csv("data/test_df_embeddings.csv", index=False)


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
    # total_df = train_df.append(validation_df).append(test_df)

    print("Fitting PCA")

    X = np.array(total_df["embeddings"].apply(lambda x: x[1:-1].split(",")))
    X = np.array([np.array(x) for x in X])
    X = X.astype(float)

    X_pca = pca.fit_transform(X)

    plt.rcParams["font.family"] = "cmr10"

    _ = plt.figure(figsize=(20, 10))

    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=total_df["color"], cmap="viridis")
    plt.xlabel("PCA 1", fontproperties="cmr10")
    plt.ylabel("PCA 2", fontproperties="cmr10")

    plt.title(
        "PCA analysis of the embeddings with color classes", fontproperties="cmr10"
    )
    handles = scatter.legend_elements()[0]
    plt.legend(handles, classes, title="Classes", prop={"family": "cmr10"})
    plt.savefig("docs/pca_labels.png", bbox_inches="tight", dpi=300)
    # plt.show()

    if save:
        # Save PCA values
        total_df["pca1"] = X_pca[:, 0]
        total_df["pca2"] = X_pca[:, 1]

        total_df.to_csv("data/total_df_embeddings.csv", index=False)


if __name__ == "__main__":
    # generate_labels()
    # labels_analysis()
    # add_embeddings()
    # pca_analysis()
    pass
