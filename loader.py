# SH-I

import numpy as np
import pandas as pd
import itertools

monomers = ['Fuc', 'GalNAc', 'Gal', 'GlcNAc', 'GlcA', 'Glc', 'KDN', 'Man', 'Neu5,9Ac2', 'Neu5Ac', 'Neu5Gc']
link_types = ['a', 'b']
occupancies = ['1', '2', '3', '4', '5', '6']
branches = ['[', ']']
sul_phos = ['3S', '4S', '6S', '6P']
linkages = ['Sp0', 'Sp8', 'Sp9', 'Sp10', 'Sp11', 'Sp12', 'Sp13', 'Sp14', 'Sp15', 'Sp16',
            'Sp17', 'Sp18', 'Sp19', 'Sp20', 'Sp21', 'Sp22', 'Sp23', 'Sp24', 'Sp25', 'MDPLys']

def vectorized_result(i):

    upper = 0.6
    lower = 0.2

    e = np.zeros((3, 1))

    if upper <= i:

        e[0] = 1.0

    elif lower <= i < upper:

        e[1] = 1.0

    elif i < lower:

        e[2] = 1.0

    return e

def encoder(glycan):

    glycan_list = []
    linker_list = []

    for unit in glycan.split('-')[:-1]:

        mono_list = []
        type_list = []
        occu_list = []
        bran_list = []
        supo_list = []

        for monomer in monomers:

            mono_list.append(unit.count(monomer + '('))

        glycan_list.append(mono_list)

        for link_type in link_types:

            type_list.append(unit.split('(')[-1].count(link_type))

        glycan_list.append(type_list)

        for occupancy in occupancies:

            occu_list.append((unit[0] + unit[-1]).count(occupancy))

        glycan_list.append(occu_list)

        for branch in branches:

            bran_list.append(unit.count(branch))

        glycan_list.append(bran_list)

        for supo in sul_phos:

            supo_list.append(unit.count(supo))

        supo_list = supo_list

        glycan_list.append(supo_list)

    for linkage in linkages:

        linker_list.append(glycan.split('-')[-1].count(linkage))

    glycan_vector = np.array(list(itertools.chain.from_iterable(glycan_list)), dtype = float)
    linker_vector = np.array(linker_list, dtype = float)
    zeroes_vector = np.zeros(920 - (len(glycan_vector) + len(linker_vector)), dtype = float)

    vector = np.concatenate((zeroes_vector, glycan_vector, linker_vector)).reshape((-1, 1)) #####

    assert len(vector) == 920, 'Vector length is not 920 bruh'
    assert np.amax(vector) < 2, 'Vector contains value more than 1 bruh'

    return vector

def load_data_old(protein, split):

    input_data = list(data['IUPAC'].apply(encoder))
    output_data = list(data.iloc[:, protein].apply(vectorized_result))

    if split == len(data['IUPAC']):

        training_data = zip(input_data, output_data)
        test_data = zip(input_data, output_data)

    else:

        training_data = zip(input_data[:split], output_data[:split])
        test_data = zip(input_data[split:], output_data[split:])

    return (training_data, test_data)

def load_data(protein, split):

    i1, i2 = int((5 * split) / 2), 5

    data = pd.read_csv('data/lectins_norm.csv')
    data = data.round(decimals = 4)
    data = data.sort_values(protein, ascending = False)

    input_data = list(data['IUPAC'])
    output_data = list(data[protein])

    if split == 0:

        training_data = zip(input_data, output_data)
        test_data = zip(input_data, output_data)

    else:

        input_strata = input_data[:i1:i2] + input_data[-i1::i2]
        output_strata = output_data[:i1:i2] + output_data[-i1::i2]

        training_data = zip([i for i in input_data if not i in input_strata or input_strata.remove(i)],
                            [i for i in output_data if not i in output_strata or output_strata.remove(i)])

        test_data = zip(input_data[:i1:i2] + input_data[-i1::i2],
                        output_data[:i1:i2] + output_data[-i1::i2])

    return (training_data, test_data)

def load_encoded(protein, split = 10):

    training_data, test_data = load_data(protein, split)

    training_input, training_output = [], []
    test_input, test_output = [], []

    for x, y in training_data:

        training_input.append(encoder(x))
        training_output.append(vectorized_result(y))

    for x, y in test_data:

        test_input.append(encoder(x))
        test_output.append(vectorized_result(y))

    training_encoded = zip(training_input, training_output)
    test_encoded = zip(test_input, test_output)

    return (training_encoded, test_encoded)

def classify(array):

    n = np.argmax(array)

    if n == 0:

        return 'high'

    elif n == 1:

        return 'medium'

    elif n == 2:

        return 'low'

def save_loaded(protein, split):

    training_data, test_data = load_data(protein, split)

    x_list = []
    y_list = []

    for x, y in test_data:

        x_list.append(x)
        y_list.append(y)

    test = pd.DataFrame()

    test['x'] = x_list
    test['y'] = y_list

    test.to_csv('test.csv', index = False)
