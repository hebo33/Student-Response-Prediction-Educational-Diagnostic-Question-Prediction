import numpy as np

from item_response import irt, sigmoid

from utils import (
    evaluate,
    load_public_test_csv,
    load_train_csv,
    load_valid_csv,
)


def bootstrap_sample(data, seed):
    """Create one bootstrap sample of the training data.

    :param data: Dictionary containing user_id,
                 question_id, and is_correct.
    :param seed: Random seed
    :return: Bootstrap training data
    """

    rng = np.random.default_rng(seed)

    num_records = len(
        data["is_correct"]
    )

    sampled_indices = rng.choice(
        num_records,
        size=num_records,
        replace=True,
    )

    bootstrap_data = {
        key: np.asarray(values)[sampled_indices].tolist()
        for key, values in data.items()
    }

    return bootstrap_data

def predict_probabilities(data, theta, beta):
    """Predict the probability of a correct response."""

    user_id = np.asarray(
        data["user_id"],
        dtype=int
    )

    question_id = np.asarray(
        data["question_id"],
        dtype=int
    )

    return sigmoid(
        theta[user_id] - beta[question_id]
    )

def ensemble_predict(data, models):
    """Average the probabilities from all base models."""

    model_predictions = []

    for theta, beta in models:
        probabilities = predict_probabilities(
            data,
            theta,
            beta,
        )

        model_predictions.append(
            probabilities
        )

    return np.mean(
        np.asarray(model_predictions),
        axis=0,
    )

def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    learning_rate = 0.0025
    iterations = 50

    seeds = [1, 2, 3]

    models = []

    for model_number, seed in enumerate(
        seeds,
        start=1
    ):
        print(
            "\nTraining base model {}...".format(
                model_number
            )
        )

        bootstrap_data = bootstrap_sample(
            train_data,
            seed
        )

        theta, beta, val_acc_lst, _, _ = irt(
            bootstrap_data,
            val_data,
            lr=learning_rate,
            iterations=iterations,
        )

        models.append(
            (theta, beta)
        )

        print(
            "Base model {} final validation "
            "accuracy: {:.4f}".format(
                model_number,
                val_acc_lst[-1],
            )
        )
    
    val_predictions = ensemble_predict(
        val_data,
        models
    )

    test_predictions = ensemble_predict(
        test_data,
        models
    )

    val_accuracy = evaluate(
        val_data,
        val_predictions
    )

    test_accuracy = evaluate(
        test_data,
        test_predictions
    )

    print(
    "Ensemble validation accuracy: {:.6f}".format(
        val_accuracy
        )
    )

    print(
    "Ensemble test accuracy: {:.6f}".format(
        test_accuracy
        )
    )
    print(
        "\nNumber of trained base models:",
        len(models)
    )


if __name__ == "__main__":
    main()