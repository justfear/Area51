import argparse
import os
import string
import re
from collections import defaultdict
from math import log10

from enum import Enum


class MessageType(Enum):
    REGULAR = 1,
    SPAM = 2


class Counter():

    def __init__(self):
        self.counter_regular = 0
        self.counter_spam = 0

    def increment_counter(self, message_type):
        """
        Increment a word's frequency count by one, depending on whether it occurred in a regular or spam message.

        :param message_type: The message type to be parsed (MessageType.REGULAR or MessageType.SPAM)
        :return: None
        """
        if message_type == MessageType.REGULAR:
            self.counter_regular += 1
        else:
            self.counter_spam += 1


class Bayespam():

    def __init__(self):
        self.regular_list = None
        self.spam_list = None
        self.regular_results = []
        self.spam_results = []
        self.n_words_regular = 0
        self.n_words_spam = 0
        self.probability_regular = 0
        self.probability_spam = 0
        self.conditional_probabilities = {}
        self.vocab = {}

    def list_dirs(self, path):
        """
        Creates a list of both the regular and spam messages in the given file path.

        :param path: File path of the directory containing either the training or test set
        :return: None
        """
        # Check if the directory containing the data exists
        if not os.path.exists(path):
            print("Error: directory %s does not exist." % path)
            exit()

        regular_path = os.path.join(path, 'regular')
        spam_path = os.path.join(path, 'spam')

        # Create a list of the absolute file paths for each regular message
        # Throws an error if no directory named 'regular' exists in the data folder
        try:
            self.regular_list = [os.path.join(regular_path, msg) for msg in os.listdir(regular_path)]
        except FileNotFoundError:
            print("Error: directory %s should contain a folder named 'regular'." % path)
            exit()

        # Create a list of the absolute file paths for each spam message
        # Throws an error if no directory named 'spam' exists in the data folder
        try:
            self.spam_list = [os.path.join(spam_path, msg) for msg in os.listdir(spam_path)]
        except FileNotFoundError:
            print("Error: directory %s should contain a folder named 'spam'." % path)
            exit()

    def read_messages(self, message_type, testing=False):
        """
        Parse all messages in either the 'regular' or 'spam' directory. Each token is stored in the vocabulary,
        together with a frequency count of its occurrences in both message types.
        Converts all words to lower case and strips them of any punctuation, spacing and numerals.
        Discards any word that has less than four letters.

        :param message_type: The message type to be parsed (MessageType.REGULAR or MessageType.SPAM)
                :param testing:
        :return: A list containing all pre-known words in
        """
        msg_index = 0
        probability_spam = self.probability_spam
        probability_regular = self.probability_regular

        if message_type == MessageType.REGULAR:
            message_list = self.regular_list
        elif message_type == MessageType.SPAM:
            message_list = self.spam_list
        else:
            message_list = []
            print("Error: input parameter message_type should be MessageType.REGULAR or MessageType.SPAM")
            exit()

        for msg in message_list:
            try:
                # Make sure to use latin1 encoding, otherwise it will be unable to read some of the messages
                with open(msg, 'r', encoding='latin1') as f:
                    # Loop through each line in the message
                    for line in f:
                        # Split the string on the space character, resulting in a list of tokens
                        split_line = line.split(" ")
                        # Loop through the tokens
                        for idx in range(len(split_line)):
                            token = split_line[idx]
                            ## Convert characters to lower case, remove punctuations, and remove digits
                            token = "".join([char.lower() for char in token if char not in string.punctuation
                                             and not char.isdigit() and not re.search("[\\\\\\s]", token)])
                            ## Encode to ascii and decode to utf-8 to remove hexadecimal numbers
                            token = token.encode('ascii', errors='ignore')
                            token = token.decode('utf-8')
                            ## Ensure we only add words with at least four letters
                            if len(token) >= 4:
                                ## Ensure the algorithm only learns when training.
                                ## Can allow it to learn while testing but would require supervision
                                if not testing:
                                    ## Increase the count of regular words or spam words, depending on message type
                                    if message_type == MessageType.REGULAR:
                                        self.n_words_regular += 1
                                    elif message_type == MessageType.SPAM:
                                        self.n_words_spam += 1
                                    if token in self.vocab.keys():
                                        # If the token is already in the vocab, retrieve its counter
                                        counter = self.vocab[token]
                                    else:
                                        # Else: initialize a new counter
                                        counter = Counter()
                                    # Increment the token's counter by one and store in the vocab
                                    counter.increment_counter(message_type)
                                    self.vocab[token] = counter
                                else:
                                    ## If testing, check that the token is in the vocab
                                    if token in self.vocab:
                                        ## If token is in the vocab, then multiply its conditional probability to the
                                        ## other two respective probabilities (logP(Regular), logP(Spam)
                                        probability_regular *= log10(self.conditional_probabilities[token]['regular'])
                                        probability_spam *= log10(self.conditional_probabilities[token]['spam'])
                    f.close()
            except Exception as e:
                print("Error while reading message %s: " % msg, e)
                exit()
            ## Compare P(regular|msg) vs P(spam|msg)
            ## If P(spam|msg) > P(regular|msg) then we add the value True to the correct list depending on message type
            if is_spam(probability_regular, probability_spam):
                if message_type == MessageType.REGULAR:
                    self.regular_results.insert(msg_index, True)
                else:
                    self.spam_results.insert(msg_index, True)
            else:
                if message_type == MessageType.REGULAR:
                    self.regular_results.insert(msg_index, False)
                else:
                    self.spam_results.insert(msg_index, False)
            ## Increment msg_index by one to differentate between messages
            msg_index += 1
            ## Reset the two probability variables to the original logP(Regular) and logP(Spam)
            probability_spam = self.probability_spam
            probability_regular = self.probability_regular

    def print_vocab(self):
        """
        Print each word in the vocabulary, plus the amount of times it occurs in regular and spam messages.

        :return: None
        """
        for word, counter in self.vocab.items():
            # repr(word) makes sure that special characters such as \t (tab) and \n (newline) are printed.
            print("%s | In regular: %d | In spam: %d" % (repr(word), counter.counter_regular, counter.counter_spam))

    def write_vocab(self, destination_fp, sort_by_freq=False):
        """
        Writes the current vocabulary to a separate .txt file for easier inspection.
        Computes conditional probabilities for both regular and spam words.

        :param destination_fp: Destination file path of the vocabulary file
        :param n_regular: The number of occurrences of regular words
        :param n_spam: The number of occurrences of spam words
        :param sort_by_freq: Set to True to sort the vocab by total frequency (descending order)
        :return: A 2-D dict containing all words as entries and their conditional probabilities (regular and spam)
        """
        ## Initialize a nested dict (2-Dimensional dict) using the defaultdict() function
        conditional_probabilities = defaultdict(dict)
        total_words = self.n_words_spam + self.n_words_regular

        if sort_by_freq:
            vocab = sorted(self.vocab.items(), key=lambda x: x[1].counter_regular + x[1].counter_spam, reverse=True)
            vocab = {x[0]: x[1] for x in vocab}
        else:
            vocab = self.vocab

        try:
            with open(destination_fp, 'w', encoding="latin1") as f:
                for word, counter in vocab.items():
                    # repr(word) makes sure that special  characters such as \t (tab) and \n (newline) are printed.
                    f.write("%s | In regular: %d | In spam: %d\n" % (
                        repr(word), counter.counter_regular, counter.counter_spam), )
                    ## If we have a 0 probability, replace it with an estimate
                    if counter.counter_regular == 0:
                        conditional_regular = 0.00000000001 / total_words
                        conditional_spam = counter.counter_spam / self.n_words_spam
                    elif counter.counter_spam == 0:
                        conditional_regular = counter.counter_regular / self.n_words_regular
                        conditional_spam = 0.000000000001 / total_words
                    ## Else compute the conditional probabilities normally
                    else:
                        conditional_regular = counter.counter_regular / self.n_words_regular
                        conditional_spam = counter.counter_spam / self.n_words_spam
                    ## Add the regular and spam conditional probabilities to the dict
                    conditional_probabilities[word]['regular'] = conditional_regular
                    conditional_probabilities[word]['spam'] = conditional_spam
                f.close()
        except Exception as e:
            print("An error occurred while writing the vocab to a file: ", e)

        return conditional_probabilities

    def test_data(self):
        ## Read and evaluate the regular test messages (testing mode: on)
        self.read_messages(MessageType.REGULAR, True)
        ## Read and evaluate the spam test messages (testing mode: on)
        self.read_messages(MessageType.SPAM, True)
        ## Compute the confusion matrix from the results


def is_spam(probability_regular, probability_spam):
    """
    Checks whether a given message is spam or regular through probability comparison.

    :param probability_regular: The number of occurrences of regular words
    :param probability_spam: The number of occurrences of spam words
    :return: True if the computed probabilities point to a spam mail, False if they point to a regular mail
    """
    return probability_spam > probability_regular



def main():
    # We require the file paths of the training and test sets as input arguments (in that order)
    # The argparse library helps us cleanly parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', type=str,
                        help='File path of the directory containing the training data')
    parser.add_argument('test_path', type=str,
                        help='File path of the directory containing the test data')
    args = parser.parse_args()

    # Read the file path of the folder containing the training set from the input arguments
    train_path = args.train_path
    ## Read the file path of the folder containing the test set from the input arguments
    test_path = args.test_path

    # Initialize a Bayespam object
    bayespam = Bayespam()
    # Initialize a list of the regular and spam message locations in the training folder
    bayespam.list_dirs(train_path)

    # Parse the messages in the regular message directory
    ## Store the sum of total regular words
    bayespam.read_messages(MessageType.REGULAR)
    # Parse the messages in the spam message directory
    ## Store the sum of total spam words
    bayespam.read_messages(MessageType.SPAM)

    ## Compute the number of regular mail messages
    n_regular_messages = len(bayespam.regular_list)
    ## Compute the number of spam mail messages
    n_spam_messages = len(bayespam.spam_list)
    ## Compute the total number of messages
    total = n_regular_messages + n_spam_messages
    ## Compute the (log) probability that a message is regular
    bayespam.probability_regular = log10(n_regular_messages / total)
    ## Compute the (log) probability that a message is spam
    bayespam.probability_spam = log10(n_spam_messages / total)
    ## Write each word and their occurrence in both spam and regular mail
    ## Store the conditional probability that a word is in either spam or regular mail in a dict
    bayespam.conditional_probabilities = bayespam.write_vocab(destination_fp="vocab.txt")

    ## Initialize a list of the regular and spam message locations in the test folder
    bayespam.list_dirs(test_path)

    ## Prompt the program in classifying the testing set for both spam and regular messages
    bayespam.test_data()

    regular_t = bayespam.regular_results.count(True)
    regular_f = bayespam.regular_results.count(False)
    spam_t = bayespam.spam_results.count(True)
    spam_f = bayespam.spam_results.count(False)

    print("                   Predicted regular             Predicted spam\n")
    print("Actually Regular:   ", regular_f,"                       ",regular_t, "\n")
    print("Actually Spam:      ", spam_f   ,"                      ",spam_t, "\n")
    print("Accuracy rate: ", 100 * (regular_f + spam_t) / (regular_f + regular_t + spam_t + spam_f), "%")

    """
    Now, implement the follow code yourselves:
    5) Bayes rule must be applied on new messages, followed by argmax classification
    6) Errors must be computed on the test set (FAR = false accept rate (misses), FRR = false reject rate (false alarms))
    7) Improve the code and the performance (speed, accuracy)
    
    Use the same steps to create a class BigramBayespam which implements a classifier using a vocabulary consisting of bigrams
    """
if __name__ == "__main__":
    main()
