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

    :return: The 2-D equivalent of the 1-D index
    """
    return idx // n_rows, idx % n_rows


def find_1D_index(row_idx, col_idx, n):
    """
    Finds the 1-D index equivalent of a given two-dimensional index in an n * n matrix
    :param n: The number of rows/columns in the matrix
    :param row_idx: The given index for the rows of a 2D matrix
    :param col_idx: The given index for the columns of the 2D matrix

    :return: The 1-D equivalent of the 2-D index
    """
    return (n * row_idx) + col_idx


def update_prototype(old_prototype, eta, input_vector):
    """
    Performs the update equation for a given prototype vector, learning rate value and input vector in order to make
    the prototype vector more similar.
    :param old_prototype: The specified old prototype vector pre-update
    :param eta: The given learning rate value
    :param input_vector: The input vector to which we are assimilating the prototye

    :return: The newly updated prototype vector
    """
    a = [x * (1 - eta) for x in old_prototype]
    b = [y * eta for y in input_vector]
    return list(map(operator.add, a, b))


class Cluster:
    """This class represents the clusters, it contains the
    prototype (the mean of all it's members) and memberlists with the
    ID's (which are Integer objects) of the datapoints that are member
    of that cluster."""

    def __init__(self, traindata, idx, random_values):
        ## Step 1 Initialise clusters randomly from the data:
        self.current_members = set([i for i in range(random_values[idx], random_values[idx + 1])])
        self.prototype = self.compute_prototype(traindata)

    def compute_prototype(self, traindata):
        """
        Computes the prototype vector (mean of all member vectors) based on the memberset of a given cluster
        :param traindata: The training data from which to take the member vectors

        :return: The prototype vector for the specified cluster
        """
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
    def __init__(self, n, epochs, traindata, testdata, dim, clients, requests):
        self.n = n
        self.epochs = epochs  ## t_max
        self.traindata = traindata
        self.testdata = testdata
        self.requests = requests
        self.clients = clients
        self.dim = dim

        self.random_values = self.create_random_clusters()
        # A 2-dimensional list of clusters. Size == N x N
        self.clusters = [[Cluster(traindata, find_1D_index(row_idx, col_idx, self.n), self.random_values)
                          for col_idx in range(n)] for row_idx in range(n)]
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
            # Step 4: All nodes within the neighbourhood of the BMU are changed,
            # you don't have to use distance relative learning.
            self.find_closest_or_in_radius(eta, radius, self.bmu_matrix, self.clusters, True)
            ## Step 5: Calculate the new learning rate and radius
            radius = self.initial_radius * (1 - (epoch / self.epochs))
            eta = self.initial_learning_rate * (1 - (epoch / self.epochs))
            self.bmu_matrix.clear()

    def test(self):
        ## Iterate along all clients. Assumption: the same clients are in the same order as in the testData
        useful_prefetched_urls = 0
        non_useful_prefetched_urls = 0
        for client in self.clients:
            ## For each client find the cluster of which it is a member and
            ## get the actual testData (the vector) of this client
            cluster, test_data_vector = self.find_cluster(client)
            for request, prediction  in zip(test_data_vector, cluster.prototype):
                ## Count number of useful requests
                if prediction > self.prefetch_threshold and request == 1:
                    useful_prefetched_urls += 1
                ## Count number of non-useful requests
                elif prediction > self.prefetch_threshold and request == 0:
                    non_useful_prefetched_urls += 1
        ## Set the global variables hitrate and accuracy to their appropriate value
        self.accuracy = useful_prefetched_urls / (useful_prefetched_urls + non_useful_prefetched_urls)
        self.hitrate = (useful_prefetched_urls + non_useful_prefetched_urls) / (len(self.requests) * len(self.testdata))

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

    def find_cluster(self, client):
        idx = self.clients.index(client)
        for row in self.clusters:
            for cluster in row:
                if idx in cluster.current_members:
                    return cluster, self.testdata[idx]

    def find_closest_or_in_radius(self, eta, radius, one_d_matrix, two_d_matrix, neighbours):
        """
        Computes the BMU for any given vector or updates prototypes in the radius of a BMU.
        :param eta: The learning rate value of the algorithm
        :param radius: The radius within which vectors are considered neighbors of the BMU
        :param one_d_matrix: The training data if computing BMU, otherwise the BMU matrix if updating neighbours
        :param two_d_matrix: An n * n matrix of clusters
        :param neighbors: True if finding/updating neighbours, False if computing BMU

        :return: None
        """
        for vector in one_d_matrix:
            distance_matrix = []
            for row in two_d_matrix:
                for element in row:
                    if neighbours:
                        distance_matrix.append(distance(vector.prototype, element.prototype))
                    else:
                        distance_matrix.append(distance(vector, element.prototype))
            if neighbours:
                self.update_neighbours([idx for idx in range(len(distance_matrix)) if distance_matrix[idx] < radius],
                                       eta,
                                       self.traindata[one_d_matrix.index(vector)])
            else:
                best_idx = distance_matrix.index(min(distance_matrix))
                idx_1, idx_2 = find_2D_index(best_idx, self.n)
                self.bmu_matrix.append(two_d_matrix[idx_1][idx_2])

    def update_neighbours(self, neighbour_list, eta, vector):
        """
        Updates all neighbours in a given list of neighbours for a given data vector and learning rate value
        :param neighbour_list: The specified list of neighbours
        :param eta: The specified learning rate value
        :param vector: The specified data vector

        :return: None
        """
        for idx in neighbour_list:
            idx_1, idx_2 = find_2D_index(idx, self.n)
            neighbour = self.clusters[idx_1][idx_2]
            update_prototype(neighbour.prototype, eta, vector)

    def create_random_clusters(self):
        """
        Creates a randomized reference list which can be used to partition the data into n * n clusters

        :return: None
        """
        ## Create a list containing the ID of all vectors randomly split in K clusters
        random_values = random.sample(range(1, len(self.traindata) - 1), (self.n * self.n) - 1)
        ## Add 0 and maximum index as points of reference
        random_values.append(0)
        random_values.append(len(self.traindata))
        ## Sort values in ascending order
        random_values.sort()

        return random_values
