import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch
import matplotlib.pyplot as plt # Add on
import copy # Add on

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)


def load_data(base_path="./data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        hidden = torch.sigmoid(self.g(inputs))
        out = torch.sigmoid(self.h(hidden))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_objectives = []
    valid_objectives = []

    for epoch in range(0, num_epoch):
        model.train()
        train_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[nan_mask] = output[nan_mask]

            # loss = torch.sum((output - target) ** 2.0) # (d)
            loss = torch.sum((output - target) ** 2.0) + (lamb / 2) * model.get_weight_norm() # (e)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_loss = evaluate_loss(model, zero_train_data, valid_data)
        valid_acc = evaluate(model, zero_train_data, valid_data)

        train_objectives.append(train_loss)
        valid_objectives.append(valid_loss)

        print(
            "Epoch: {}\tTraining Cost: {:.6f}\t"
            "Validation Cost: {:.6f}\tValid Acc: {:.6f}".format(
                epoch + 1,
                train_loss,
                valid_loss,
                valid_acc
            )
        )

    return train_objectives, valid_objectives
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for i, user_id in enumerate(valid_data["user_id"]):
            inputs = train_data[user_id].unsqueeze(0)
            output = model(inputs)

            question_id = valid_data["question_id"][i]
            guess = output[0][question_id].item() >= 0.5

            if guess == valid_data["is_correct"][i]:
                correct += 1

            total += 1

    return correct / float(total)

def evaluate_loss(model, zero_train_data, valid_data):
    """Evaluate the squared-error objective on the validation data.
    """
    # Evaluate model
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for i, user_id in enumerate(valid_data["user_id"]):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            output = model(inputs)

            question_id = valid_data["question_id"][i]
            target = float(valid_data["is_correct"][i])

            prediction = output[0][question_id]
            total_loss += (prediction - target) ** 2

    return total_loss.item()


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    num_question = zero_train_matrix.shape[1]

    ###############################################################
    # (c) Tune k
    ###############################################################
    k_values = [10, 50, 100, 200, 500]

    # Fixed optimization hyperparameters
    lr = 0.01
    num_epoch = 25
    lamb = 0

    best_k = None
    best_valid_accuracy = -1

    for k in k_values:
        print(f"\nTraining k = {k}")

        model = AutoEncoder(num_question, k)

        train_objectives, valid_objectives = train(
            model,
            lr,
            lamb,
            train_matrix,
            zero_train_matrix,
            valid_data,
            num_epoch
        )

        valid_accuracy = evaluate(
            model,
            zero_train_matrix,
            valid_data
        )

        print("Validation Accuracy:", valid_accuracy)

        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_k = k

    ###############################################################
    # Tune learning rate
    ###############################################################
    learning_rates = [0.005, 0.01, 0.05]

    best_lr = None
    best_valid_accuracy = -1

    for lr in learning_rates:

        print(f"\nTraining lr = {lr}")

        model = AutoEncoder(num_question, best_k)

        train(
            model,
            lr,
            lamb,
            train_matrix,
            zero_train_matrix,
            valid_data,
            num_epoch
        )

        valid_accuracy = evaluate(
            model,
            zero_train_matrix,
            valid_data
        )

        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_lr = lr

    ###############################################################
    # Tune epochs
    ###############################################################
    epoch_values = [10, 25, 50]

    best_epoch = None
    best_valid_accuracy = -1

    for epoch in epoch_values:

        print(f"\nTraining epochs = {epoch}")

        model = AutoEncoder(num_question, best_k)

        train_objectives, valid_objectives = train(
            model,
            best_lr,
            lamb,
            train_matrix,
            zero_train_matrix,
            valid_data,
            epoch
        )

        valid_accuracy = evaluate(
            model,
            zero_train_matrix,
            valid_data
        )

        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_epoch = epoch

            best_model = model
            best_train_objectives = train_objectives
            best_valid_objectives = valid_objectives

    ###############################################################
    # Results for (c)
    ###############################################################
    print("\nBest hyperparameters")
    print("k =", best_k)
    print("learning rate =", best_lr)
    print("epochs =", best_epoch)
    print("validation accuracy =", best_valid_accuracy)

    ###############################################################
    # (d)
    ###############################################################
    test_accuracy = evaluate(
        best_model,
        zero_train_matrix,
        test_data
    )

    print("Test accuracy =", test_accuracy)

    epochs = range(1, best_epoch + 1)

    plt.figure()
    plt.plot(epochs, best_train_objectives, label="Training Objective")
    plt.plot(epochs, best_valid_objectives, label="Validation Objective")
    plt.xlabel("Epoch")
    plt.ylabel("Objective")
    plt.legend()
    plt.grid(True)
    plt.savefig("autoencoder_objectives.png")
    plt.show()

    ###############################################################
    # (e)
    ###############################################################
    lambda_values = [0.001, 0.01, 0.1, 1]

    best_lambda = None
    best_regularized_valid = -1

    for lamb in lambda_values:

        print(f"\nTraining lambda = {lamb}")

        model = AutoEncoder(num_question, best_k)

        train(
            model,
            best_lr,
            lamb,
            train_matrix,
            zero_train_matrix,
            valid_data,
            best_epoch
        )

        valid_accuracy = evaluate(
            model,
            zero_train_matrix,
            valid_data
        )

        if valid_accuracy > best_regularized_valid:
            best_regularized_valid = valid_accuracy
            best_lambda = lamb
            best_regularized_model = model

    regularized_test = evaluate(
        best_regularized_model,
        zero_train_matrix,
        test_data
    )

    print("\nBest lambda =", best_lambda)
    print("Regularized validation accuracy =", best_regularized_valid)
    print("Regularized test accuracy =", regularized_test)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
