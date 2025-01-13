import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import random


def load_data(embedding_file, label_file):
    """Load embeddings and labels from CSV files."""
    embeddings = pd.read_csv(embedding_file)
    labels = pd.read_csv(label_file)
    return embeddings, labels


# Embedding merge functions
def average_pooling(embeddings):
    """Average Pooling."""
    return np.mean(embeddings, axis=0)


def max_pooling(embeddings):
    """Max Pooling."""
    return np.max(embeddings, axis=0)


def softmax_pooling(embeddings):
    """Softmax Pooling."""
    weights = np.exp(embeddings) / np.sum(np.exp(embeddings), axis=0)
    return np.sum(embeddings * weights, axis=0)


def logistic_regression_pooling(embeddings):
    """Logistic Regression-based Pooling."""
    lr = LogisticRegression()
    labels = [0] * len(embeddings) + [1]
    embeddings = np.vstack((embeddings, np.zeros_like(embeddings[0])))
    lr.fit(embeddings, labels)
    return lr.coef_[0]


# Extend with additional pooling strategies as needed...


# Bag creation
def create_bags_for_split(embeddings, labels, bag_size, enforce_balance, split=None):
    """Create bags for a given split or dataset."""
    bags = []
    labels_0 = labels[labels["label"] == 0]
    labels_1 = labels[labels["label"] == 1]

    def create_bag(data_0, data_1):
        bag = []
        label = 0
        if len(data_1) > 0:
            label = 1
            n_1 = random.randint(1, min(bag_size, len(data_1)))
            bag += random.sample(data_1, n_1)
        n_0 = bag_size - len(bag)
        if n_0 > 0:
            bag += random.sample(data_0, n_0)
        return bag, label

    while len(labels_0) >= bag_size or len(labels_1) >= 1:
        if enforce_balance and len(labels_1) > 0:
            bag, bag_label = create_bag(labels_0.tolist(), labels_1.tolist())
        else:
            bag = random.sample(labels.index.tolist(), min(bag_size, len(labels)))
            bag_label = 1 if any(labels.loc[i, "label"] == 1 for i in bag) else 0
        bag_embeddings = embeddings.loc[bag]
        bags.append((bag_embeddings, bag_label, split))
        labels = labels.drop(bag)
    return bags


def merge_embeddings(bags, merge_method):
    """Merge embeddings in each bag using the selected method."""
    merge_function = {
        "average": average_pooling,
        "max": max_pooling,
        "softmax": softmax_pooling,
        "logistic": logistic_regression_pooling,
    }[merge_method]

    merged_bags = []
    for embeddings, label, split in bags:
        merged_embedding = merge_function(embeddings.values[:, 1:])  # Exclude the filename column
        merged_bags.append((merged_embedding, label, split))
    return merged_bags


def save_bags(output_file, bags, output_format):
    """Save bags to output CSV file."""
    with open(output_file, "w") as f:
        if output_format == "single_column":
            f.write("Embedding Label Split\n")
            for embedding, label, split in bags:
                split_value = split if split is not None else ""
                f.write(f"{' '.join(map(str, embedding))} {label} {split_value}\n")
        else:
            header = [f"Dim_{i}" for i in range(len(bags[0][0]))] + ["Label", "Split"]
            f.write(",".join(header) + "\n")
            for embedding, label, split in bags:
                split_value = split if split is not None else ""
                f.write(",".join(map(str, embedding)) + f",{label},{split_value}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiple Instance Learning Bag Creator")
    parser.add_argument("--embeddings", required=True, help="Input CSV file with embeddings")
    parser.add_argument("--labels", required=True, help="Input CSV file with labels and splits")
    parser.add_argument("--use_split", action="store_true", help="Use the split column if present")
    parser.add_argument("--bag_size", type=int, required=True, help="Bag size (static number or range)")
    parser.add_argument("--merge_method", required=True, choices=["average", "max", "softmax", "logistic"],
                        help="Method to merge embeddings")
    parser.add_argument("--enforce_balance", action="store_true", help="Enforce balanced bags")
    parser.add_argument("--output_format", required=True, choices=["single_column", "multi_column"],
                        help="Output format")
    parser.add_argument("--output", required=True, help="Output CSV file")
    args = parser.parse_args()

    embeddings, labels = load_data(args.embeddings, args.labels)
    bags = create_bags(embeddings, labels, args.use_split, args.bag_size, args.enforce_balance)
    merged_bags = merge_embeddings(bags, args.merge_method)
    save_bags(args.output, merged_bags, args.output_format)
