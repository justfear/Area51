"""kmeans.py"""
import random
import operator
import math
import copy


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
        self.beginning = True

    def update_member_sets(self):
        ## Move the current member set to previous, and clear the current member set
        self.previous_members = copy.deepcopy(self.current_members)
        ## It does not matter if the membership has stabilized since both sets would be the same anyways
        self.current_members.clear()


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


def prototype_computation(cluster, vector):
    """
    Adds a given vector to the prototype variable of a given cluster for the prototype calculation
    :param cluster: The specified cluster
    :param vector: The specified vector
    :return: None
    """
    ## If we are iterating through the first vector ID in the cluster, make it equal to the prototype variable
    if cluster.prototype_start:
        cluster.prototype = vector
        cluster.prototype_start = False
    ## Otherwise sum the vectors together
    else:
        cluster.prototype = list(map(operator.add, cluster.prototype, vector))


class KMeans:
    def __init__(self, k, traindata, testdata, dim, clients, requests):
        self.traindata = traindata
        self.testdata = testdata
        self.requests = requests
        self.clients = clients
        self.dim = dim

        # Threshold above which the corresponding html is prefetched
        self.prefetch_threshold = 0.5
        # An initialized list of k clusters
        self.clusters = [Cluster(dim) for _ in range(k)]

        # The accuracy and hitrate are the performance metrics (i.e. the results)
        self.accuracy = 0
        self.hitrate = 0

    def train(self):
        ## Step 1: Select an initial random partitioning with k clusters
        self.create_random_clusters(1)
        i = 0
        ## Repeat loop while membership has not stabilized
        ## Set prototype_start boolean to True
        ## Finish prototype computation (divide the summed vectors)
        while not self.check_equal():
            print(i)
            i += 1
            print(self.clusters[0].previous_members)
            print(self.clusters[1].previous_members)
            ## Step 2: Generate a new partition by assigning each datapoint to its closest cluster center
            ## Simultaneously begin the computation of the prototype
            self.compare_distances()

    def test(self):

        ## iterate along all clients. Assumption: the same clients are in the same order as in the testData
        useful_prefetched_urls = 0
        non_useful_prefetched_urls = 0
        for client in self.clients:
            ## For each client find the cluster of which it is a member and
            ## get the actual testData (the vector) of this client
            cluster, test_data_vector = self.find_cluster(client)
            for request, prediction in zip(test_data_vector, cluster.prototype):
                ## Count number of useful requests
                if prediction > self.prefetch_threshold and request == 1:
                    useful_prefetched_urls += 1
                ## Count number of non-useful requests
                elif prediction > self.prefetch_threshold and request == 0:
                    non_useful_prefetched_urls += 1
        ## Set the global variables hitrate and accuracy to their appropriate value
        self.accuracy = (useful_prefetched_urls / (useful_prefetched_urls + non_useful_prefetched_urls)) / len(self.clients)
        self.hitrate = (useful_prefetched_urls / len(self.requests)) / len(self.clients)





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

    def find_cluster(self, client):
        idx = self.clients.index(client)
        for cluster in self.clusters:
            if idx in cluster.current_members:
                return cluster, self.testdata[idx]


    def create_random_clusters(self, idx):
        """
        Partitions all data vectors into randomly split clusters, computes prototypes simultaneously

        :param idx: Always starts at 1, a point of reference for the indexing of numbers in the random_values list
        :return: None
        """
        ## Create a list containing the ID of all vectors randomly split in K clusters
        random_values = random.sample(range(1, len(self.traindata) - 1), len(self.clusters) - 1)
        ## Add 0 and maximum index as points of reference
        random_values.append(0)
        random_values.append(len(self.traindata))
        ## Sort values in ascending order
        random_values.sort()
        print("Random values:", random_values)

        for cluster in self.clusters:
            for i in range(random_values[idx - 1], random_values[idx]):
                ## Add the datapoint ID to our cluster's current_members
                cluster.current_members.add(i)
                prototype_computation(cluster, self.traindata[i])
            cluster.prototype = [x / len(cluster.current_members) for x in cluster.prototype]
            idx += 1

    def compare_distances(self):
        """
        Finds cluster closest to each vector and assigns vectors to said cluster.
        Simultaneously sums up assigned vectors to their respective cluster prototype.

        :return: None
        """
        for vector in self.traindata:
            distance_matrix = []
            for cluster in self.clusters:
                ## Compute and store a distance between vector and prototype in a list
                distance_matrix.append(distance(vector, cluster.prototype))
            ## Find the index (in the matrix) of the cluster closest to the vector
            ## The index in the matrix is the same as the index in the cluster list
            cluster_idx = distance_matrix.index(min(distance_matrix))
            ## Add the vector's index to the current_members set of the aforementioned cluster
            self.clusters[cluster_idx].current_members.add(self.traindata.index(vector))
            ## Add the vector to the prototype computation of said cluster
            prototype_computation(self.clusters[cluster_idx], vector)

    def check_equal(self):
        """
        Resets prototype booleans, finalizes the prototype calculation
        and checks for membership stabilization for each cluster.

        :return: True if membership has stabilized, otherwise False
        """
        check = True
        for cluster in self.clusters:
            ## If we just initialized the clusters, then instantly return False, and update the beginning boolean
            if cluster.beginning:
                for cluster in self.clusters:
                    cluster.update_member_sets()
                    cluster.beginning = False
                return False
            ## Reset the prototype_start boolean
            cluster.prototype_start = True
            ## Finalize the calculation of the average of the summed vectors in each cluster (prototype calculation)
            cluster.prototype = [x / len(cluster.current_members) for x in cluster.prototype]
            ## Sort both sets to make sure everything is in the same order
            cluster.previous_members = set(sorted(cluster.previous_members))
            cluster.current_members = set(sorted(cluster.current_members))
            ## Check whether membership has stabilized in all clusters
            for current, previous in zip(cluster.current_members, cluster.previous_members):
                if current != previous:
                    check = False
            if not check:
                cluster.update_member_sets()
        return check
