"""Two-parameter item response theory model for CSC311 Project Part B.

This file extends the one-parameter IRT model from Part A by learning a
positive discrimination parameter alpha_j for every question:

    p(c_ij = 1) = sigmoid(alpha_j * (theta_i - beta_j)).

We optimize gamma_j = log(alpha_j), so alpha_j = exp(gamma_j) is always
positive. Optional L2 regularization is included for the Part B experiments.
"""

from utils import load_public_test_csv, load_train_csv, load_valid_csv

import csv
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    """Return a numerically stable sigmoid of a scalar or NumPy array."""
    return np.exp(-np.logaddexp(0, -np.asarray(x)))


def _as_arrays(data):
    """Convert a response dictionary to aligned NumPy arrays."""
    return (
        np.asarray(data["user_id"], dtype=int),
        np.asarray(data["question_id"], dtype=int),
        np.asarray(data["is_correct"], dtype=float),
    )


def predict_probabilities(data, theta, beta, gamma):
    """Predict probabilities for all student-question pairs in data."""
    user_id, question_id, _ = _as_arrays(data)
    alpha = np.exp(gamma[question_id])
    score = alpha * (theta[user_id] - beta[question_id])
    return sigmoid(score)


def neg_log_likelihood(
    data,
    theta,
    beta,
    gamma,
    regularization=0.0,
):
    """Return the regularized negative log-likelihood of the 2PL model.

    The unregularized loss is

        sum(log(1 + exp(z_ij)) - c_ij * z_ij),

    where z_ij = exp(gamma_j) * (theta_i - beta_j).
    """
    user_id, question_id, is_correct = _as_arrays(data)
    alpha = np.exp(gamma[question_id])
    score = alpha * (theta[user_id] - beta[question_id])

    nll = np.sum(np.logaddexp(0, score) - is_correct * score)
    penalty = 0.5 * regularization * (
        np.sum(theta ** 2)
        + np.sum(beta ** 2)
        + np.sum(gamma ** 2)
    )
    return float(nll + penalty)


def update_parameters(
    data,
    learning_rate,
    theta,
    beta,
    gamma,
    regularization=0.0,
):
    """Perform one simultaneous gradient-ascent update.

    We maximize log-likelihood minus an L2 penalty. Since
    alpha_j = exp(gamma_j), the log-likelihood gradients are

        dL/dtheta_i = sum_j alpha_j * (c_ij - p_ij)
        dL/dbeta_j  = sum_i alpha_j * (p_ij - c_ij)
        dL/dgamma_j = sum_i alpha_j * (theta_i - beta_j)
                            * (c_ij - p_ij).
    """
    user_id, question_id, is_correct = _as_arrays(data)

    alpha = np.exp(gamma[question_id])
    ability_difference = theta[user_id] - beta[question_id]
    probability = sigmoid(alpha * ability_difference)
    residual = is_correct - probability

    theta_gradient = np.zeros_like(theta)
    beta_gradient = np.zeros_like(beta)
    gamma_gradient = np.zeros_like(gamma)

    np.add.at(
        theta_gradient,
        user_id,
        alpha * residual,
    )
    np.add.at(
        beta_gradient,
        question_id,
        -alpha * residual,
    )
    np.add.at(
        gamma_gradient,
        question_id,
        alpha * ability_difference * residual,
    )

    theta_gradient -= regularization * theta
    beta_gradient -= regularization * beta
    gamma_gradient -= regularization * gamma

    new_theta = theta + learning_rate * theta_gradient
    new_beta = beta + learning_rate * beta_gradient
    new_gamma = gamma + learning_rate * gamma_gradient

    # This wide safety range prevents numerical overflow without forcing the
    # learned discrimination values to be close to one.
    new_gamma = np.clip(new_gamma, -3.0, 3.0)
    return new_theta, new_beta, new_gamma


def evaluate(data, theta, beta, gamma):
    """Return classification accuracy using a probability threshold of 0.5."""
    probabilities = predict_probabilities(data, theta, beta, gamma)
    predictions = probabilities >= 0.5
    answers = np.asarray(data["is_correct"], dtype=bool)
    return float(np.mean(predictions == answers))


def train_2pl(
    train_data,
    val_data,
    learning_rate,
    iterations,
    regularization=0.0,
    verbose=True,
):
    """Train 2PL IRT and return the best validation checkpoint and history."""
    num_users = max(
        max(train_data["user_id"]),
        max(val_data["user_id"]),
    ) + 1
    num_questions = max(
        max(train_data["question_id"]),
        max(val_data["question_id"]),
    ) + 1

    theta = np.zeros(num_users)
    beta = np.zeros(num_questions)
    gamma = np.zeros(num_questions)  # alpha = exp(0) = 1

    history = {
        "train_nll": [],
        "val_nll": [],
        "val_accuracy": [],
    }

    best_val_accuracy = -np.inf
    best_iteration = 0
    best_parameters = None

    for iteration in range(iterations + 1):
        # Do not include the training regularizer in the reported likelihoods;
        # this makes train and validation curves directly comparable.
        train_nll = neg_log_likelihood(
            train_data,
            theta,
            beta,
            gamma,
        )
        val_nll = neg_log_likelihood(
            val_data,
            theta,
            beta,
            gamma,
        )
        val_accuracy = evaluate(
            val_data,
            theta,
            beta,
            gamma,
        )

        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)
        history["val_accuracy"].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_iteration = iteration
            best_parameters = (
                theta.copy(),
                beta.copy(),
                gamma.copy(),
            )

        if verbose:
            print(
                "Iteration: {} | Train NLL: {:.4f} | "
                "Validation NLL: {:.4f} | Validation Accuracy: {:.4f}".format(
                    iteration,
                    train_nll,
                    val_nll,
                    val_accuracy,
                )
            )

        if iteration < iterations:
            theta, beta, gamma = update_parameters(
                train_data,
                learning_rate,
                theta,
                beta,
                gamma,
                regularization,
            )

    best_theta, best_beta, best_gamma = best_parameters
    return (
        best_theta,
        best_beta,
        best_gamma,
        best_iteration,
        history,
    )


def plot_training_curves(history, train_size, val_size, filename):
    """Save average train/validation log-likelihood curves."""
    iteration_values = np.arange(len(history["train_nll"]))
    average_train_log_likelihood = (
        -np.asarray(history["train_nll"]) / train_size
    )
    average_val_log_likelihood = (
        -np.asarray(history["val_nll"]) / val_size
    )

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    ax.plot(
        iteration_values,
        average_train_log_likelihood,
        label="Training",
    )
    ax.plot(
        iteration_values,
        average_val_log_likelihood,
        label="Validation",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Log-Likelihood")
    ax.set_title("2PL IRT Training and Validation Log-Likelihood")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_discrimination_example(filename):
    """Illustrate how alpha changes a 2PL question response curve."""
    theta_values = np.linspace(-3, 3, 300)
    beta = 0.0

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    for alpha in (0.5, 1.0, 2.0):
        probabilities = sigmoid(alpha * (theta_values - beta))
        ax.plot(
            theta_values,
            probabilities,
            label=f"alpha = {alpha}",
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.axvline(beta, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Student Ability (theta)")
    ax.set_ylabel("Probability of a Correct Response")
    ax.set_title("Effect of Question Discrimination in 2PL IRT")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_validation_grid(results, learning_rates, regularizations, filename):
    """Plot validation accuracy for every hyperparameter combination."""
    accuracy_matrix = np.empty(
        (len(learning_rates), len(regularizations))
    )
    for row_index, learning_rate in enumerate(learning_rates):
        for column_index, regularization in enumerate(regularizations):
            matching_result = next(
                result
                for result in results
                if result["learning_rate"] == learning_rate
                and result["regularization"] == regularization
            )
            accuracy_matrix[row_index, column_index] = matching_result[
                "validation_accuracy"
            ]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    image = ax.imshow(
        accuracy_matrix,
        cmap="Blues",
        vmin=accuracy_matrix.min() - 0.0005,
        vmax=accuracy_matrix.max() + 0.0005,
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(regularizations)))
    ax.set_xticklabels([str(value) for value in regularizations])
    ax.set_yticks(np.arange(len(learning_rates)))
    ax.set_yticklabels([str(value) for value in learning_rates])
    ax.set_xlabel("L2 Regularization Strength")
    ax.set_ylabel("Learning Rate")
    ax.set_title("2PL Validation Accuracy")

    for row_index in range(len(learning_rates)):
        for column_index in range(len(regularizations)):
            ax.text(
                column_index,
                row_index,
                f"{accuracy_matrix[row_index, column_index]:.4f}",
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(image, ax=ax, label="Validation Accuracy")
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(
    baseline_validation_accuracy,
    baseline_test_accuracy,
    modified_validation_accuracy,
    modified_test_accuracy,
    filename,
):
    """Plot validation/test accuracy of the Part A and Part B models."""
    labels = ["Original 1PL", "Modified 2PL"]
    validation_values = [
        baseline_validation_accuracy,
        modified_validation_accuracy,
    ]
    test_values = [
        baseline_test_accuracy,
        modified_test_accuracy,
    ]
    positions = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    validation_bars = ax.bar(
        positions - width / 2,
        validation_values,
        width,
        label="Validation",
    )
    test_bars = ax.bar(
        positions + width / 2,
        test_values,
        width,
        label="Test",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("Original and Modified IRT Accuracy")
    ax.set_ylim(0.69, 0.715)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(validation_bars, fmt="%.4f", padding=3)
    ax.bar_label(test_bars, fmt="%.4f", padding=3)
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_experiments(train_data, val_data, test_data):
    """Run the controlled Part B grid search and save all results.

    Hyperparameters are selected using validation accuracy only. The test set
    is evaluated once, after the best validation configuration is fixed.
    """
    learning_rates = [0.001, 0.0025, 0.005, 0.01]
    regularizations = [0.0, 1.0, 3.0, 10.0]
    iterations = 100
    results = []

    best_result = None
    for learning_rate in learning_rates:
        for regularization in regularizations:
            theta, beta, gamma, best_iteration, history = train_2pl(
                train_data,
                val_data,
                learning_rate=learning_rate,
                iterations=iterations,
                regularization=regularization,
                verbose=False,
            )
            alpha = np.exp(gamma)
            result = {
                "learning_rate": learning_rate,
                "regularization": regularization,
                "best_iteration": best_iteration,
                "training_accuracy": evaluate(
                    train_data,
                    theta,
                    beta,
                    gamma,
                ),
                "validation_accuracy": evaluate(
                    val_data,
                    theta,
                    beta,
                    gamma,
                ),
                "average_train_nll": neg_log_likelihood(
                    train_data,
                    theta,
                    beta,
                    gamma,
                ) / len(train_data["is_correct"]),
                "average_validation_nll": neg_log_likelihood(
                    val_data,
                    theta,
                    beta,
                    gamma,
                ) / len(val_data["is_correct"]),
                "minimum_alpha": float(alpha.min()),
                "median_alpha": float(np.median(alpha)),
                "maximum_alpha": float(alpha.max()),
                "theta": theta,
                "beta": beta,
                "gamma": gamma,
                "history": history,
            }
            results.append(result)

            if (
                best_result is None
                or result["validation_accuracy"]
                > best_result["validation_accuracy"]
            ):
                best_result = result

            print(
                "lr={} | lambda={} | best iteration={} | "
                "validation accuracy={:.4f}".format(
                    learning_rate,
                    regularization,
                    best_iteration,
                    result["validation_accuracy"],
                )
            )

    test_accuracy = evaluate(
        test_data,
        best_result["theta"],
        best_result["beta"],
        best_result["gamma"],
    )
    best_result["test_accuracy"] = test_accuracy

    csv_fields = [
        "learning_rate",
        "regularization",
        "best_iteration",
        "training_accuracy",
        "validation_accuracy",
        "average_train_nll",
        "average_validation_nll",
        "minimum_alpha",
        "median_alpha",
        "maximum_alpha",
    ]
    with open(
        "part_b_experiment_results.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as result_file:
        writer = csv.DictWriter(result_file, fieldnames=csv_fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in csv_fields})

    plot_validation_grid(
        results,
        learning_rates,
        regularizations,
        "part_b_validation_grid.png",
    )
    return best_result


def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    best_result = run_experiments(
        train_data,
        val_data,
        test_data,
    )

    print("Selected learning rate:", best_result["learning_rate"])
    print("Selected regularization:", best_result["regularization"])
    print("Best iteration:", best_result["best_iteration"])
    print(
        "Final validation accuracy:",
        best_result["validation_accuracy"],
    )
    print("Final test accuracy:", best_result["test_accuracy"])
    print("Minimum learned alpha:", best_result["minimum_alpha"])
    print("Median learned alpha:", best_result["median_alpha"])
    print("Maximum learned alpha:", best_result["maximum_alpha"])

    plot_training_curves(
        best_result["history"],
        len(train_data["is_correct"]),
        len(val_data["is_correct"]),
        "irt_2pl_log_likelihood_curve.png",
    )
    plot_discrimination_example(
        "irt_2pl_discrimination_curves.png",
    )
    plot_model_comparison(
        baseline_validation_accuracy=0.7084391758396839,
        baseline_test_accuracy=0.7005362686988428,
        modified_validation_accuracy=best_result["validation_accuracy"],
        modified_test_accuracy=best_result["test_accuracy"],
        filename="part_b_model_comparison.png",
    )


if __name__ == "__main__":
    main()
