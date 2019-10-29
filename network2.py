# SH-I

import json
import random
import sys
import numpy as np
import pickle

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):

        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):

        return (a - y) * sigmoid_prime(z)

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):

        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):

        return (a - y)

class Network(object):

    def __init__(self, sizes, cost = CrossEntropyCost):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):

        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):

        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):

        for b, w in zip(self.biases, self.weights):

            a = sigmoid(np.dot(w, a)+b)

        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data = None,
            monitor_evaluation_cost = False,
            monitor_evaluation_accuracy = False,
            monitor_training_cost = False,
            monitor_training_accuracy = False,
            early_stopping_n = 0):

        best_accuracy = 1

        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:

            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        best_accuracy = 0
        no_accuracy_change = 0

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in range(epochs):

            random.shuffle(training_data)

            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:

                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))

            print("Epoch %s training complete" % j)

            if monitor_training_cost:

                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)

                print("Cost on training data: {}".format(cost))

            if monitor_training_accuracy:

                accuracy = self.accuracy(training_data)
                training_accuracy.append(accuracy)

                print("Accuracy on training data: {} / {}".format(accuracy, n))

            if monitor_evaluation_cost:

                cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(cost)

                print("Cost on evaluation data: {}".format(cost))

            if monitor_evaluation_accuracy:

                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)

                print("Accuracy on evaluation data: {} / {}".format(accuracy, n_data))

            if early_stopping_n > 0:

                if accuracy > best_accuracy:

                    best_accuracy = accuracy
                    no_accuracy_change = 0

                    print("Early-stopping: Best so far {}".format(best_accuracy))

                else:

                    no_accuracy_change += 1

                if (no_accuracy_change == early_stopping_n):

                    print("Early-stopping: No accuracy change in last epochs: {}".format(early_stopping_n))

                    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

            if j == epochs - 1:

                result_data = self.results(evaluation_data)

                with open('results.pkl', 'wb') as f:

                    pickle.dump(result_data, f)

                print('Results saved')

        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:

            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):

            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):

            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def accuracy(self, data, convert = True):

        if convert:

            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:

            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)

        return result_accuracy

    def total_cost(self, data, lmbda, convert = True):

        cost = 0.0

        for x, y in data:

            a = self.feedforward(x)
            cost += self.cost.fn(a, y)/len(data)
            cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)

        return cost

    def save(self, filename):

        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}

        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    def results(self, test_data):

        test_results = [(self.feedforward(x), y)
                        for (x, y) in test_data]

        return test_results

def load(filename):

    f = open(filename, "r")
    data = json.load(f)
    f.close()

    cost = CrossEntropyCost
    net = Network(data["sizes"], cost = cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]

    return net

def sigmoid(z):

    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):

    return sigmoid(z) * (1 - sigmoid(z))
