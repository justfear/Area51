"""kmeans.py"""
import random
import operator
import math


class Cluster:
    """This class represents the clusters, it contains the
    prototype (the mean of all it's members) and memberlists with the
    ID's (which are Integer objects) of the datapoints that are member
    of that cluster. You also want to remember the previous members so
    you can check if the clusters are stable."""

    def __init__(self, dim):
        self.prototype = [0.0 for _ in range(dim)]
        self.current_members = set()
        self.previous_members = set()
        self.prototype_start = True


class KMeans:
    def __init__(self, k, traindata, testdata, dim):
        self.traindata = traindata
        self.testdata = testdata
        self.dim = dim

        # Threshold above which the corresponding html is prefetched
        self.prefetch_threshold = 0.5
        # An initialized list of k clusters
        self.clusters = [Cluster(dim) for _ in range(k)]

        # The accuracy and hitrate are the performance metrics (i.e. the results)
        self.accuracy = 0
        self.hitrate = 0

    def train(self):
        # Step 1: Select an initial random partioning with k clusters
        self.create_random_clusters(1)

        while not self.check_equal():
            # Step 2: Generate a new partition by assigning each datapoint to its closest cluster center
            self.compare_distances()
            # Step 3: recalculate cluster centers




    def test(self):
        # iterate along all clients. Assumption: the same clients are in the same order as in the testData
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
        for i, cluster in enumerate(self.clusters):
            print("Members cluster", i, ":", cluster.current_members)

    def print_prototypes(self):
        for i, cluster in enumerate(self.clusters):
            print("Prototype cluster", i, ":", cluster.prototype)

    def create_random_clusters(self, idx):
        """
        Partitions all data vectors into randomly split clusters, computes prototypes simultaneously

        :param idx: Always starts at 1, Point of reference for indexing of numbers in the random_values list
        :return: None
        """
        # Create a list containing the ID of all vectors randomly split in K clusters
        random_values = random.sample(range(1, len(self.traindata) - 1), len(self.clusters))
        # Add 0 and 70 for ease of performance
        random_values.append(0)
        random_values.append(len(self.traindata))
        # Sort values in ascending order
        random_values.sort()

        for cluster in self.clusters:
            # If we are iterating through the first vector ID in the cluster, make it equal to the prototype variable
            for i in range(random_values[idx - 1], random_values[idx]):
                if i == random_values[idx - 1]:
                    cluster.prototype = self.traindata[i]
                # Otherwise add any subsequent vector's elements together
                else:
                    cluster.prototype = list(map(operator.add, cluster.prototype, self.traindata[i]))
                # Add the datapoint ID to our cluster's current_members
                cluster.previous_members.add(i)
            # Compute the mean of each value of all vectors added together in a cluster
            cluster.prototype = [x / (random_values[idx] - random_values[idx - 1]) for x in cluster.prototype]
            idx += 1

    def compare_distances(self):
        for vector in self.traindata:
            distance_matrix = []
            for cluster in self.clusters:
                distance_matrix.append(distance(vector, cluster.prototype))
            cluster_idx = distance_matrix.index(max(distance_matrix))
            self.clusters[cluster_idx].current_members.add(self.traindata.index(vector))
            if self.clusters[cluster_idx].prototype_start:
                self.clusters[cluster_idx].prototype = vector
                self.clusters[cluster_idx].prototype_start = False
            else:
                self.clusters[cluster_idx].prototype = list(
                    map(operator.add, self.clusters[cluster_idx].prototype, vector))

    def check_equal(self):
        for cluster in self.clusters:
            for a, b in zip(cluster.current_members, cluster.previous_members):
                if a != b:
                    return False
        return True

def distance(vector, prototype):
    total = 0
    for x, p in zip(vector, prototype):
        total += pow(x - p, 2)
        return math.sqrt(total)
