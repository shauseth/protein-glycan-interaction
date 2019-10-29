# SH-I

import loader
import network2
import analysis
import overfitting

split = 0 # size of the test set
num_epochs = 100 # number of epochs
plot = False
protein = 'DC_SIGN'

training_data, test_data = loader.load_encoded(protein, split = split)

net = network2.Network([920, 100, 3], cost = network2.CrossEntropyCost())

data = net.SGD(training_data, num_epochs, 10, 0.5,
              evaluation_data = test_data, lmbda = 0,
              monitor_evaluation_cost = True,
              monitor_evaluation_accuracy = True,
              monitor_training_cost = True,
              monitor_training_accuracy = True)

if plot:

    overfitting.make_plots(data, num_epochs, training_set_size = 606 - split)

    analysis.plot_confusion()

net.save('network.json')

loader.save_loaded(protein, split)
