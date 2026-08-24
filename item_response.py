from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt



def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    user_id = np.asarray(
        data["user_id"],
        dtype=int
    )

    question_id = np.asarray(
        data["question_id"],
        dtype=int
    )

    is_correct = np.asarray(
        data["is_correct"],
        dtype=float
    )
    x = theta[user_id] - beta[question_id]

    log_likelihood = np.sum(
        is_correct * x - np.logaddexp(0, x)
        )

    return -log_likelihood
    


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    user_id = np.asarray(
        data["user_id"],
        dtype=int
    )

    question_id = np.asarray(
        data["question_id"],
        dtype=int
    )

    is_correct = np.asarray(
        data["is_correct"],
        dtype=float
    )


    theta_gradient = np.zeros_like(theta)
    beta_gradient = np.zeros_like(beta)
    
    probability = sigmoid(
        theta[user_id] - beta[question_id]
    )

    np.add.at(
        theta_gradient,
        user_id,
        is_correct - probability
    )

    np.add.at(
        beta_gradient,
        question_id,
        probability - is_correct
    )

    theta = theta + lr * theta_gradient
    beta = beta + lr * beta_gradient
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    num_users = max(
        max(data["user_id"]),
        max(val_data["user_id"])
    ) + 1

    num_questions = max(
        max(data["question_id"]),
        max(val_data["question_id"])
    ) + 1

    theta = np.zeros(num_users)
    beta = np.zeros(num_questions)

    val_acc_lst = []
    train_nll_lst = []
    val_nll_lst = []

    for i in range(iterations + 1):
        train_nll = neg_log_likelihood(data, theta=theta, beta=beta)

        val_nll = neg_log_likelihood(
            val_data,
            theta,
            beta
        )

        val_accuracy = evaluate(
            val_data,
            theta,
            beta
        )

        train_nll_lst.append(train_nll)
        val_nll_lst.append(val_nll)
        val_acc_lst.append(val_accuracy)

        print(
            "Iteration: {} | Train NLL: {:.4f} | "
            "Validation NLL: {:.4f} | "
            "Validation Accuracy: {:.4f}".format(
                i,
                train_nll,
                val_nll,
                val_accuracy
            )
        )
        if i == iterations:
            break
        theta, beta = update_theta_beta(data, lr, theta, beta)

    
    return theta, beta, val_acc_lst, train_nll_lst, val_nll_lst


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    learning_rate = 0.0025
    iterations = 50

    theta, beta, val_acc_lst, train_nll_lst, val_nll_lst = irt(
        train_data,
        val_data,
        lr=learning_rate,
        iterations=iterations,
    )

    iteration_values = np.arange(iterations + 1)

    average_train_log_likelihood = (
        -np.asarray(train_nll_lst)
        / len(train_data["is_correct"])
    )

    average_val_log_likelihood = (
        -np.asarray(val_nll_lst)
        / len(val_data["is_correct"])
    )

    fig, ax = plt.subplots(
        figsize=(8, 5.5),
        constrained_layout=True
    )

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
    ax.set_title(
        "IRT Training and Validation Log-Likelihood",
        pad=12
    )
    ax.legend()
    ax.grid(alpha=0.3)

    fig.savefig(
        "irt_log_likelihood_curve.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Selected learning rate:", learning_rate)
    print("Selected number of iterations:", iterations)
    print("Final validation accuracy:", val_acc_lst[-1])

    final_test_accuracy = evaluate(
        test_data,
        theta,
        beta
    )   

    print(
    "Final test accuracy:",
    final_test_accuracy
    )

    sorted_question_ids = np.argsort(beta)

    selected_question_ids = [
        sorted_question_ids[0],
        sorted_question_ids[len(sorted_question_ids) // 2],
        sorted_question_ids[-1],
    ]

    difficulty_labels = [
        "Easy",
        "Medium",
        "Difficult"
    ]

    theta_values = np.linspace(
        theta.min() - 0.5,
        theta.max() + 0.5,
        300,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5.5),
        constrained_layout=True
    )

    for question_id, difficulty_label in zip(
        selected_question_ids,
        difficulty_labels,
    ):
        probabilities = sigmoid(
            theta_values - beta[question_id]
        )

        ax.plot(
            theta_values,
            probabilities,
            label=(
                f"{difficulty_label}: Question {question_id} "
                f"(beta={beta[question_id]:.3f})"
            ),
        )

    ax.axhline(
        0.5,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    ax.set_xlabel("Student Ability (theta)")
    ax.set_ylabel("Probability of a Correct Response")
    ax.set_title("IRT Question Response Curves")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.savefig(
        "irt_question_response_curves.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    for difficulty_label, question_id in zip(
        difficulty_labels,
        selected_question_ids,
    ):
        print(
        "{} question: ID {}, beta = {:.4f}".format(
            difficulty_label,
            question_id,
            beta[question_id],
        )
    )

if __name__ == "__main__":
    main()
