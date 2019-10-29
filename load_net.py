# SH-I

import loader
import network2
import numpy as np

glycan = input('Enter Glycan IUPAC: ')

net = network2.load('net.json')
gly_enc = loader.encoder(glycan)

binding = loader.classify(net.feedforward(gly_enc))

print('This glycan has a ' + binding + ' binding to DC_SIGN.')
