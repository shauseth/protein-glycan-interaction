# SH-I

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix

def plot_confusion(filename = 'confusion'):

    with open('results.pkl', 'rb') as f:

        results = pickle.load(f)

    y_pred = []
    y_actu = []

    for x, y in results:

        y_pred.append(np.argmax(y))
        y_actu.append(np.argmax(x))

    confusion = confusion_matrix(y_actu, y_pred, [0, 1, 2])

    fig, ax = plt.subplots()

    plt.imshow(confusion, cmap = 'Blues')
    plt.xticks(np.arange(3), ['high', 'med', 'low'], rotation = 'vertical')
    plt.yticks(np.arange(3), ['high', 'med', 'low'])
    plt.xlabel('output')
    plt.ylabel('desired output')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    threshold = confusion.max() / 2.0

    for i in range(3):

        for j in range(3):

            if confusion[i, j] > threshold:

                text = ax.text(j, i, confusion[i, j], ha = 'center', va = 'center', color = 'w')

            else:

                text = ax.text(j, i, confusion[i, j], ha = 'center', va = 'center')

    plt.show()

def plot_heat():

    with open('results.pkl', 'rb') as f:

        results = pickle.load(f)

    for i, data in enumerate(results):

        filename = 'results/' + str(i + 1)

        x, y = data

        plt.subplot(1, 2, 1)
        plt.imshow(x, cmap = 'Blues')
        plt.title('output')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(y, cmap = 'Blues')
        plt.title('desired output')
        plt.axis('off')

        plt.savefig(filename + '.png', bbox_inches = 'tight')
