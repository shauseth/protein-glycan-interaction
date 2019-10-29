# SH-I

import matplotlib.pyplot as plt
import numpy as np

def make_plots(data, num_epochs,
               training_cost_xmin=200,
               test_accuracy_xmin=200,
               test_cost_xmin=0,
               training_accuracy_xmin=0,
               training_set_size=506):

    test_cost, test_accuracy, training_cost, training_accuracy = data

    # plot_training_cost(training_cost, num_epochs, training_cost_xmin)
    # plot_test_cost(test_cost, num_epochs, test_cost_xmin)
    plot_overlay(test_accuracy, training_accuracy, num_epochs,
                 training_set_size)

def plot_training_cost(training_cost, num_epochs, training_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(training_cost)),
            training_cost,
            color='#2A6EA6')
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the training data')
    plt.show()

def plot_test_cost(test_cost, num_epochs, test_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(test_cost)),
            test_cost,
            color='#2A6EA6')
    ax.set_xlim([test_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the test data')
    plt.show()

def plot_overlay(test_accuracy, training_accuracy, num_epochs,
                 training_set_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(test_accuracy)),
            [accuracy * 100.0 / (606 - training_set_size)
             for accuracy in test_accuracy],
            color='#2A6EA6',
            label="Accuracy (%) on the test data")
    ax.plot(np.arange(len(training_accuracy)),
            [accuracy * 100.0 / training_set_size
             for accuracy in training_accuracy],
            color='#FFA933',
            label="Accuracy (%) on the training data")
    ax.grid(True)
    ax.set_xlabel('Epoch')
    plt.legend(loc="lower right")
    #plt.ylim(bottom=0)
    plt.show()
