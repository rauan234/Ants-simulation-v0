from math import tanh
import random
import numpy as np


class Neuron:
    def __init__(self, bias):
        self.bias = bias  # tuple of two floats; see update_activation

        self.axons = []  # each axon is a tuple of the form (receiving_neuron, weight)

        self.activation = 0  # the current activation of this neuron; can have any real number value
        self.incoming_signals = []  # the list of signals sent by other neurons to this neuron via axons

    def register_input(self, input_signal):
        # input signal is of the form (axon_weight, sender_neuron_activation)
        self.incoming_signals.append(input_signal)

    def update_activation(self):
        incoming = sum(list(map(lambda sg: sg[0] * sg[1], self.incoming_signals)))
        self.activation = self.bias[0] + tanh(self.bias[1] + incoming)

        self.incoming_signals = []

    def add_axon(self, new_axon):
        self.axons.append(new_axon)


class Network:
    def __init__(self, n_input_neurons, n_output_neurons):
        self.n_input_neurons, self.n_output_neurons = n_input_neurons, n_output_neurons

        self.input_neurons = [Neuron((0, 0)) for i in range(n_input_neurons)]
        self.output_neurons = [Neuron((0, 0)) for i in range(n_output_neurons)]

        self.middle_neurons = []

    def send_impulses(self):
        axon_activations = []

        for neuron in self.input_neurons + self.middle_neurons:
            for axon in neuron.axons:
                receiving_neuron, weight = axon
                activation = neuron.activation
                receiving_neuron.register_input((weight, activation))
                axon_activations.append((axon, activation))

    def update_activations(self):
        for neuron in self.middle_neurons + self.output_neurons:
            neuron.update_activation()

    def update(self, input_values):
        # input_values = [v_0, ..., v_{n-1}]
        # set the activation of the i-th input neuron to v_i
        for (neuron, activation) in zip(self.input_neurons, input_values):
            neuron.activation = activation

        # process the impulses that neurons send to each other at this time step
        self.send_impulses()

        # given the impulses that neurons have sent to each other, compute their new activations
        self.update_activations()

        return [neuron.activation for neuron in self.output_neurons]

    def __str__(self):
        all_neurons = self.input_neurons + self.middle_neurons + self.output_neurons
        indexes = {neuron: i for (i, neuron) in enumerate(all_neurons)}

        s = str(len(self.input_neurons)) + ' ' + str(len(self.middle_neurons)) + ' ' + str(len(self.output_neurons))

        for (i, neuron) in enumerate(self.input_neurons):
            s += f'\n{i}>'
            for receiving_neuron, weight in neuron.axons:
                s += ' ' + str(indexes[receiving_neuron]) + '|' + str(weight)

        for (i, neuron) in enumerate(self.middle_neurons):
            s += f'\n{i+self.n_input_neurons}> {neuron.activation} {neuron.bias[0]} {neuron.bias[1]}'
            for receiving_neuron, weight in neuron.axons:
                s += ' ' + str(indexes[receiving_neuron]) + '|' + str(weight)

        return s

    def write(self, filename):  # save self to a given text file
        with open(filename, 'w') as file:
            file.write(self.__str__())

    def add_neuron(self):  # randomly generate a neuron and add it to self
        bias = (random.normalvariate(0, 0.5), random.normalvariate(0, 0.5))
        nrn = Neuron(bias)
        nrn.activation = tanh(random.normalvariate(0, 3))
        self.middle_neurons.append(nrn)

    def add_random_axon(self):
        sender_neuron = random.choice(self.input_neurons + self.middle_neurons)

        receiver_neuron = sender_neuron
        while receiver_neuron == sender_neuron:
            receiver_neuron = random.choice(self.middle_neurons + self.output_neurons)

        weight = np.random.normal()

        sender_neuron.add_axon((receiver_neuron, weight))

    def copy(self):  # a really lazy option; remake later
        return read_network_from_lines(iter(str(self).split('\n')))


def read_network_from_lines(lines: iter):
    # read the number of input, middle, and output neurons
    n_inp, n_mid, n_out = tuple(map(int, lines.__next__().split()))

    nw = Network(n_inp, n_out)
    for i in range(n_mid):
        nw.add_neuron()

    def get_neuron_by_index(i):
        if i < n_inp:
            return nw.input_neurons[i]
        elif i < n_inp + n_mid:
            return nw.middle_neurons[i - n_inp]
        else:
            return nw.output_neurons[i - n_inp - n_mid]

    for i in range(n_inp + n_mid):
        sender_neuron = get_neuron_by_index(i)

        line = lines.__next__()
        if i < n_inp:  # if reading an input neuron
            _, *axons_spec_raw = line.split()
            # read the axons coming out of this neuron
            activation = 0  # activation of input neuron need not be read; it will be given as input to the neural net
            bias = (0, 0)  # same applies to bias
        else:  # if reading a middle neuron
            _, activation, bias0, bias1, *axons_spec_raw = line.split()
            activation = float(activation)
            bias = (float(bias0), float(bias1))
            # read the activation and bias of this middle neuron as well as the axons coming out of it
        axons_spec = [(int(p[0]), float(p[1])) for p in list(map(lambda s: s.split('|'), axons_spec_raw))]

        sender_neuron.activation = activation
        sender_neuron.bias = bias

        for (j, weight) in axons_spec:
            receiving_neuron = get_neuron_by_index(j)
            sender_neuron.add_axon((receiving_neuron, weight))

    return nw


def read_network_from_file(filename: str):
    with open(filename, 'r') as file:
        nw = read_network_from_lines(file.__iter__())

    return nw


if __name__ == '__main__':
    nw = Network(3, 2)
    for i in range(10): nw.add_neuron()
    for i in range(40): nw.add_random_axon()
    print(nw)

    print()

    nw.write('mynetwork.txt')

    nw_new = read_network_from_file('mynetwork.txt')
    print(nw_new)
