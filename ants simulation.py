import random
import numpy as np
from math import sin, cos, exp, pi, sqrt, tanh
from PIL import Image

import continuous_neural_network as nnet  # another file in this project


NumberOfScentMarkers = 3  # the number of scent markers ants can use


def hex_color_format(r, g, b):
    def f(c):
        return f'{c:x}'.zfill(2)
    return '#' + f(r) + f(g) + f(b)


def dot(u, v):
    n = len(u)
    assert len(v) == n

    out = 0
    for i in range(n):
        out += u[i] * v[i]
    return out


def signed_step(x):
    if x <= -1:
        return -1
    elif x >= 1:
        return 1
    else:  # -1 < x < 1
        return 0


class AntsParameters:
    def __init__(self, decay_coefficients, neural_net):
        assert (isinstance(decay_coefficients, tuple) and len(decay_coefficients) == NumberOfScentMarkers and
                    all(map(lambda x: 0 <= x <= 1, decay_coefficients)))
        assert isinstance(neural_net, nnet.Network)

        self.marker_decay_coefficients = decay_coefficients  # the rate of decay of various scent markers
        self.nw = neural_net

        self.speed = 3  # maximum distance the ant can move in one time step
        self.angular_speed = pi/4  # maximum angle the ant can turn by in one time step

        self.food_cost = 0  # how much food it costs to produce an ant
        self.calculate_food_cost()

    def calculate_food_cost(self):  # NEED TO UPDATE LATER
        self.food_cost = 3


def read_ants_parameters_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.__iter__()

        decay_coefs = tuple(map(float, lines.__next__().split()))
        assert len(decay_coefs) == NumberOfScentMarkers

        nw = nnet.read_network_from_lines(lines)
        assert nw.n_input_neurons == (2 + 3*NumberOfScentMarkers)
        # 3 neurons per scent type, 1 neuron for random number generator (gaussian), 1 neuron for bias (activation = 1)
        assert nw.n_output_neurons == (2 + NumberOfScentMarkers)
        # 1 neuron per scent type, 1 neuron for "forward", 1 neuron for "turn left"

    return AntsParameters(decay_coefs, nw)


class Ant:
    def __init__(self, parameters, position, orientation):
        self.parameters = parameters  # an instance of AntsParameters class

        self.nw = self.parameters.nw.copy()  # initialize a neural network to guide ant's movement

        self.position = position  # tuple of the form (x, y), where x and y are NOT necessarily integers
        self.orientation = orientation  # rotation clockwise from x-axis, measured in radians

        self.carrying_food = False  # whether the ant currently has food in its jaws
        # that it is supposed to transport to the mother colony

        self.forward = self.left = (0, 0)
        self.update_unit_vectors()

    def update_unit_vectors(self):
        self.forward = (cos(self.orientation), sin(self.orientation))  # unit vector pointing forward
        self.left = (-sin(self.orientation), cos(self.orientation))  # unit vector pointing left

    def display(self, img):
        size = 6  # the scale that is used to display the ant
        # (does not affect its behavior at all, only changes appearance on visualization)

        pixels = img.load()

        # draw a "double-ball" shape
        self_x, self_y = round(self.position[0]), round(self.position[1])
        xmin, xmax = max(0, self_x - size), min(img.size[0], self_x + size)
        ymin, ymax = max(0, self_y - size), min(img.size[1], self_y + size)
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                d = (x - self.position[0], y - self.position[1])
                u, v = dot(d, self.forward) / size, dot(d, self.left) / size
                if u**2 + (2*v)**2 <= sqrt(0.15 + abs(u - 0.35)):
                    pixels[x, y] = (255, 255, 255)

    def update(self, scent_input):  # move ant to new position, return amount of each scent marker deposited
        nw_input = [1, random.normalvariate(0, 1)]  # first bias neuron, then random number generator neuron
        for (value, d_x, d_y) in scent_input:
            d_f = dot((d_x, d_y), self.forward)
            d_l = dot((d_x, d_y), self.left)

            nw_input.append(value)  # the value of this scent at ant's position
            nw_input.append(d_f)  # the gradient of this scent along the frontal axis
            nw_input.append(d_l)  # the gradient of this scent along the side axis

        nw_output = self.nw.update(nw_input)
        desire_forward, desire_turn = nw_output[0], nw_output[1]  # desire to move forward and turn
        dl = self.parameters.speed * signed_step(desire_forward / 0.5)
        dth = self.parameters.angular_speed * min(max(2*(desire_turn - 0.5), -1), 1)
        # signed_step(x) is -1 if x <= -1, 0 if -1 < x < 1, 1 if 1 <= x
        # tanh is the hyperbolic tangent; it is also between -1 and 1, but it is continuous

        scent_output = list(map(lambda x: max(2*(x-0.5), 0), nw_output[2:]))
        # the amount of scent deposited

        self.position[0] += self.forward[0] * dl
        self.position[1] += self.forward[1] * dl
        self.orientation += dth
        self.update_unit_vectors()
        # note: checking if the ant is inside [0, xsize-1] x [0, ysize-1] is not necessary,
        # since this check will be done in Simulation.update; self does not have access to xsize and ysize anyway

        return scent_output


class AntColony:
    def __init__(self, ant_reporting_function, ants_parameters, position, radius):
        self.ant_reporting_function = ant_reporting_function  # once a new ant is produced,
        # ant_reporting_function is called to register it in some sort of database that is external to self

        self.ants_parameters = ants_parameters  # specifies the kind of ants that are produced in this colony

        self.position = position  # tuple of the form (x, y)
        self.radius = radius  # the colony is physically a circle with this radius

        self.food_surplus = 0  # once food_surplus reaches a certain threshold
        # specified by ants_parameters, a new ant is produced.
        # food comes in units of 1, so food_surplus is always an integer (also nonnegative)

    def display(self, img):
        pixels = img.load()

        self_x, self_y = round(self.position[0]), round(self.position[1])
        xmin, xmax = max(0, self_x - self.radius), min(img.size[0], self_x + self.radius)
        ymin, ymax = max(0, self_y - self.radius), min(img.size[1], self_y + self.radius)
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                if (x - self.position[0])**2 + (y - self.position[1])**2 <= self.radius**2:
                    pixels[x, y] = (255, 255, 255)

    def is_inside(self, pos):  # check if point pos = (x, y) is inside the colony
        return (pos[0] - self.position[0])**2 + (pos[1] - self.position[1])**2 <= self.radius**2

    def receive_one_unit_of_food(self):
        self.food_surplus += 1

    def produce_ant(self):
        theta = random.random() * 2*pi

        pos = [self.position[0] + self.radius * cos(theta),
               self.position[1] + self.radius * sin(theta)]
        new_ant = Ant(self.ants_parameters, pos, theta)

        self.ant_reporting_function(new_ant)

    def update(self):
        if self.food_surplus >= self.ants_parameters.food_cost:
            self.produce_ant()


class Field:
    def __init__(self, xsize, ysize):
        self.xsize = xsize
        self.ysize = ysize
        self.marker_values = np.zeros((NumberOfScentMarkers, xsize, ysize))

    def process_ant_scent_input(self, ant_position, markers_data):
        # ant at position ant_position placed q_0 of marker 0, q_1 of marker 1, ..., q_{n-1} of marker n-1,
        # where q_0, ..., q_{n-1} are given as markers_data = [q_0, ..., q_{n-1}]

        x = round(ant_position[0])
        y = round(ant_position[1])
        for i in range(NumberOfScentMarkers):
            self.marker_values[i][x][y] += markers_data[i]

    def decay_scent_markers(self, decay_coefficients):
        for i in range(NumberOfScentMarkers):
            self.marker_values[i] *= decay_coefficients[i]

    def display(self, img):
        def f(x):
            return 1 - exp(-x)

        def color_function(m):
            return (round(255 * f(m[0])), round(255 * f(m[1])), round(255 * f(m[2])))

        pixels = img.load()

        for x in range(self.xsize):
            for y in range(self.ysize):
                m = [self.marker_values[i][x][y] for i in range(NumberOfScentMarkers)]

                pixels[x, y] = color_function(m)

    def point_in_range(self, x, y):
        return 0 <= x < self.xsize and 0 <= y < self.ysize

    def get_scent_at(self, pos):  # returns [(value_0, dx_0, dy_0), ..., (value_{n-1}, dx_{n-1}, dy_{n-1})]
        # return the values and gradients of scent markers at a given (non-integer) position
        # this function uses linear model fitting; it looks at the 3x3 square surrounding the ant
        # to infer the values and gradients of the scent values
        out = []

        for i in range(NumberOfScentMarkers):
            reference_points = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    x, y = round(pos[0]) + dx, round(pos[1]) + dy
                    if self.point_in_range(x, y):
                        reference_points.append((x, y, self.marker_values[i][x][y]))

            X = [[1, p[0], p[1]] for p in reference_points]
            y = [p[2] for p in reference_points]
            coef = np.linalg.lstsq(X, y, rcond=1)[0]

            # tuple of the form (value, d_x, d_y)
            out.append((coef.dot(np.array((1, pos[0], pos[1]))), coef[1], coef[2]))

        return out


class Simulation:
    def __init__(self, ants_parameters, xsize, ysize, colony_x=None, colony_y=None, colony_r=None):
        self.ants_parameters = ants_parameters

        self.field = Field(xsize, ysize)
        self.ants = []

        if colony_x is None: colony_x = xsize/2
        if colony_y is None: colony_y = ysize/2
        if colony_r is None: colony_r = 40
        self.colony = AntColony(lambda new_ant: self.ants.append(new_ant),
                                ants_parameters, (colony_x, colony_y), colony_r)

    def update(self):
        # decay the scent markers left by ants in previous iterations
        self.field.decay_scent_markers(self.ants_parameters.marker_decay_coefficients)

        # for each ant, move it to its new position
        # and account for the scent markers it decides to place in this iteration.
        # also, see if ants brought food to the colony
        for ant in self.ants:
            # if ant went outside the field, bring it back
            ant.position[0] = min(max(ant.position[0], 0), self.field.xsize - 1)
            ant.position[1] = min(max(ant.position[1], 0), self.field.ysize - 1)

            if ant.carrying_food and self.colony.is_inside(ant.position):
                ant.carrying_food = False
                self.colony.receive_one_unit_of_food()

            scent_input = self.field.get_scent_at(ant.position)  # compute what the ant smells

            ant_initial_position = ant.position.copy()  # store ant's original position in a variable
            scent_output = ant.update(scent_input)  # see what scent markers ant deposits, update ant's position

            self.field.process_ant_scent_input(ant_initial_position, scent_output)  # place the scent markers

        # evolve the colony by one time step; namely, if there is enough food, create a new ant
        self.colony.update()

    def display(self, img):
        self.field.display(img)

        for ant in self.ants:
            ant.display(img)

        self.colony.display(img)


def main():
    xsize, ysize = 600, 400

    images = []

    ants_parameters = read_ants_parameters_from_file('ants_params.txt')
    s = Simulation(ants_parameters, xsize, ysize)

    for i in range(10):
        s.colony.produce_ant()

    img = Image.new('RGB', (xsize, ysize))
    s.display(img)
    images.append(img)
    for i in range(30):
        print(i)
        s.update()

        img = Image.new('RGB', (xsize, ysize))
        s.display(img)
        images.append(img)

    print(f'Produced {len(images)} frames')
    images[0].save('animation.gif',
                   save_all=True, append_images=images[1:], optimize=False, duration=80, loop=1)
    print('Saved successfully!')
    input()


main()
