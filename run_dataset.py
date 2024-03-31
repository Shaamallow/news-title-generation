# Run Analysis on the dataset

from functools import total_ordering
from torch import values_copy
from src.load_data import load_data
from src.load_models import device
from src.labels import labels_classification

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def generate_labels():
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
    TRAIN_PATH = "data/train_labels.csv"
    VALIDATION_PATH = "data/validation_labels.csv"
    TEST_PATH = "data/test_labels.csv"
    train_df, validation_df, test_df = load_data(TRAIN_PATH, VALIDATION_PATH, TEST_PATH)

    fix, axs = plt.subplots(1, 3, figsize=(20, 5))
    plt.rcParams["font.family"] = "cmr10"

    # pie plot
    train_df["genre"].value_counts().plot.pie(ax=axs[0], autopct="%1.1f%%")
    axs[0].set_title("Train Data", fontproperties="cmr10", fontsize=20)
    axs[0].set_ylabel("Count : " + str(len(train_df)), fontproperties="cmr10")

    validation_df["genre"].value_counts().plot.pie(ax=axs[1], autopct="%1.1f%%")
    axs[1].set_title("Validation Data", fontproperties="cmr10", fontsize=20)
    axs[1].set_ylabel("Count : " + str(len(validation_df)), fontproperties="cmr10")

    test_df["genre"].value_counts().plot.pie(ax=axs[2], autopct="%1.1f%%")
    axs[2].set_title("Test Data", fontproperties="cmr10", fontsize=20)
    axs[2].set_ylabel("Count : " + str(len(test_df)), fontproperties="cmr10")

    # Save
    plt.savefig("docs/labels_repartition.png", bbox_inches="tight", dpi=300)


def pca_analysis():
    total_df = load_data("data/total_df_embeddings.csv")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    pca = PCA(n_components=2)
    # total_df = train_df.append(validation_df).append(test_df)

    # tokenize the text
    total_df["embeddings"] = total_df["text"].apply(
        lambda x: tokenizer(x, return_tensors="pt", padding=True, truncation=True)
    )

    total_df.to_csv("data/total_df_embeddings.csv", index=False)

    X = pca.fit_transform(total_df["embeddings"].values)
    plt.scatter(X[:, 0], X[:, 1], c=total_df["genre"], cmap="viridis")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # generate_labels()
    labels_analysis()
    # pca_analysis()
