import random
import operator
import math


def distance(vector, prototype):
    """
    Calculates the Euclidean distance between a given data vector and a prototype vector
    :param vector: The specified data vector
    :param prototype: The specified prototype vector

    :return: The linear distance between the two vectors
    """
    total = 0
    for x, p in zip(vector, prototype):
        total += pow(x - p, 2)
        return math.sqrt(total)


def find_2D_index(idx, n_rows):
    """
    Finds the 2-D index equivalent of a one dimensional index in a n * n matrix
    :param idx: The one dimensional index which we want to convert to two dimensions
    :param n_rows: The number of rows (n) in an n * n matrix

    :return: The 2-D equivalent of the 1D index
    """
    return idx // n_rows, idx % n_rows


def update_prototype(old_prototype, eta, vector):
    a = [x * (1-eta) for x in old_prototype]
    b = [y * eta for y in vector]
    return list(map(operator.add, a, b))


class Cluster:
    """This class represents the clusters, it contains the
    prototype (the mean of all it's members) and memberlists with the
    ID's (which are Integer objects) of the datapoints that are member
    of that cluster."""

    def __init__(self, traindata):
        ## Step 1 Initialise clusters randomly from the data:
        self.current_members = set(random.sample(range(len(traindata) - 1), 1))
        self.prototype = self.compute_prototype(traindata)

    def compute_prototype(self, traindata):
        first = True
        prototype = []
        for idx in self.current_members:
            if first:
                prototype = traindata[idx]
                first = False
            else:
                prototype = list(map(operator.add, prototype, traindata[idx]))

        return [x / len(self.current_members) for x in prototype]


class Kohonen:
    def __init__(self, n, epochs, traindata, testdata, dim):
        self.n = n
        self.epochs = epochs  ## t_max
        self.traindata = traindata
        self.testdata = testdata
        self.dim = dim

        # A 2-dimensional list of clusters. Size == N x N
        self.clusters = [[Cluster(traindata) for _ in range(n)] for _ in range(n)]
        self.bmu_matrix = []
        # Threshold above which the corresponding html is prefetched
        self.prefetch_threshold = 0.5
        self.initial_learning_rate = 0.8  ## eta
        self.initial_radius = (n * n) / 2  ## radius
        # The accuracy and hitrate are the performance metrics (i.e. the results)
        self.accuracy = 0
        self.hitrate = 0

    def train(self):
        # Step 2: Calculate the squareSize and the learningRate, these decrease
        # linearly with the number of epochs.
        radius = self.initial_radius
        eta = self.initial_learning_rate
        # Repeat 'epochs' times:
        for epoch in range(1, self.epochs):
            ## Step 3: Every input vector is presented to the map (always in the same
            ## order) For each vector its Best Matching Unit is found, and
            ## each cluster in the vector's neighborhood is found:
            self.find_closest_or_in_radius(eta, radius, self.traindata, self.clusters, False)
            self.find_closest_or_in_radius(eta, radius, self.bmu_matrix, self.clusters, True)
            # Step 4: All nodes within the neighbourhood of the BMU are changed,
            # you don't have to use distance relative learning.

            ## Step 5: Calculate the new learning rate and radius
            radius = self.initial_radius * (1 - (self.epochs / epoch))
            eta = self.initial_learning_rate * (1 - (self.epochs / epoch))

    def test(self):
        # iterate along all clients
        # for each client find the cluster of which it is a member
        # get the actual testData (the vector) of this client
        # iterate along all dimensions
        # and count prefetched htmls
        # count number of hits
        # count number of requests
        # set the global variables hitrate and accuracy to their appropriate value
        pass

    def print_test(self):
        print("Prefetch threshold =", self.prefetch_threshold)
        print("Hitrate:", self.hitrate)
        print("Accuracy:", self.accuracy)
        print("Hitrate+Accuracy =", self.hitrate + self.accuracy)

    def print_members(self):
        for i in range(self.n):
            for j in range(self.n):
                print("Members cluster", (i, j), ":", self.clusters[i][j].current_members)

    def print_prototypes(self):
        for i in range(self.n):
            for j in range(self.n):
                print("Prototype cluster", (i, j), ":", self.clusters[i][j].prototype)

    def find_closest_or_in_radius(self, eta, radius, one_d_matrix, two_d_matrix, neighbors):
        for vector in one_d_matrix:
            distance_matrix = []
            for row in two_d_matrix:
                for element in row:
                    if neighbors:
                        distance_matrix.append(distance(vector.prototype, element.prototype))
                    else:
                        distance_matrix.append(distance(vector, element.prototype))
            if neighbors:
                self.update_neighbours([idx for idx in range(len(distance_matrix)) if distance_matrix[idx] < radius],
                                       eta,
                                       self.traindata[one_d_matrix.index(vector)])
            else:
                best_idx = distance_matrix.index(min(distance_matrix))
                idx_1, idx_2 = find_2D_index(best_idx, self.n)
                self.bmu_matrix.append(two_d_matrix[idx_1][idx_2])

    def update_neighbours(self, neighbour_list, eta, vector):
        for idx in neighbour_list:
            idx_1, idx_2 = find_2D_index(idx, self.n)
            neighbour = self.clusters[idx_1][idx_2]
            update_prototype(neighbour.prototype, eta, vector)
