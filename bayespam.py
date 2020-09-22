import argparse
import os
import string
import re
from collections import defaultdict

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

    def read_messages(self, message_type):
        """
        Parse all messages in either the 'regular' or 'spam' directory. Each token is stored in the vocabulary,
        together with a frequency count of its occurrences in both message types.
        :param message_type: The message type to be parsed (MessageType.REGULAR or MessageType.SPAM)
        :return: The total number of words contained in the messages specified
        """
        total = 0
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
                f = open(msg, 'r', encoding='latin1')

                # Loop through each line in the message
                for line in f:
                    # Split the string on the space character, resulting in a list of tokens
                    split_line = line.split(" ")
                    # Loop through the tokens
                    for idx in range(len(split_line)):
                        token = split_line[idx]
                        ## Convert characters to lower case, remove punctuations, and remove digits
                        token = "".join([char.lower() for char in token if char not in string.punctuation
                                         and not char.isdigit() and not re.search("[\\\\\s]", token)])
                        ## Remove any words with fewer than four letters
                        token = '' if len(token) < 4 else token
                        if token in self.vocab.keys():
                            # If the token is already in the vocab, retrieve its counter
                            counter = self.vocab[token]
                        else:
                            # Else: initialize a new counter
                            counter = Counter()

                        # Increment the token's counter by one and store in the vocab
                        counter.increment_counter(message_type)
                        ## If the token is empty, we don't add it, else it will return true if it is non-empty
                        if token:
                            total += 1
                            self.vocab[token] = counter

            except Exception as e:
                print("Error while reading message %s: " % msg, e)
                exit()

        return total

    def print_vocab(self):
        """
        Print each word in the vocabulary, plus the amount of times it occurs in regular and spam messages.

        :return: None
        """
        for word, counter in self.vocab.items():
            # repr(word) makes sure that special characters such as \t (tab) and \n (newline) are printed.
            print("%s | In regular: %d | In spam: %d" % (repr(word), counter.counter_regular, counter.counter_spam))

    def write_vocab(self, destination_fp, n_regular, n_spam, sort_by_freq=False):
        """
        Writes the current vocabulary to a separate .txt file for easier inspection.

        :param destination_fp: Destination file path of the vocabulary file
        :param n_regular: The number of occurrences of regular words
        :param n_spam: The number of occurrences of spam words
        :param sort_by_freq: Set to True to sort the vocab by total frequency (descending order)
        :return: A 2-D dictionary containing all words as entries and their conditional probabilities (regular and spam)
        """
        repertory = defaultdict(dict)

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
                        conditional_regular = 1 / (n_regular + n_spam)
                        conditional_spam = counter.counter_spam / n_spam
                    elif counter.counter_spam == 0:
                        conditional_regular = counter.counter_regular / n_regular
                        conditional_spam = 1 / (n_regular + n_spam)
                    else:
                        conditional_regular = counter.counter_regular / n_regular
                        conditional_spam = counter.counter_spam / n_spam
                    repertory[word]['regular'] = conditional_regular
                    repertory[word]['spam'] = conditional_spam
                f.close()
        except Exception as e:
            print("An error occurred while writing the vocab to a file: ", e)

        return repertory


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

    # Initialize a Bayespam object
    bayespam = Bayespam()
    # Initialize a list of the regular and spam message locations in the training folder
    bayespam.list_dirs(train_path)

    # Parse the messages in the regular message directory
    n_words_regular = bayespam.read_messages(MessageType.REGULAR)
    # Parse the messages in the spam message directory
    n_words_spam = bayespam.read_messages(MessageType.SPAM)

    # bayespam.print_vocab()
    repertory = bayespam.write_vocab(destination_fp="vocab.txt", n_regular=n_words_regular, n_spam=n_words_spam)

    print("N regular messages: ", len(bayespam.regular_list))
    print("N spam messages: ", len(bayespam.spam_list))
    total = len(bayespam.regular_list) + len(bayespam.spam_list)
    print("Total amount of massages", total)
    print("Probability for regular", len(bayespam.regular_list) / total)
    print("Probability for spam", len(bayespam.spam_list) / total)
    print("Sum of total regular words in the vocabulary ", n_words_regular)
    print("Sum of total regular words in the vocabulary ", n_words_spam)

    """
    Now, implement the follow code yourselves:
    1) A priori class probabilities must be computed from the number of regular and spam messages
    2) The vocabulary must be clean: punctuation and digits must be removed, case insensitive
    3) Conditional probabilities must be computed for every word
    4) Zero probabilities must be replaced by a small estimated value
    5) Bayes rule must be applied on new messages, followed by argmax classification
    6) Errors must be computed on the test set (FAR = false accept rate (misses), FRR = false reject rate (false alarms))
    7) Improve the code and the performance (speed, accuracy)
    
    Use the same steps to create a class BigramBayespam which implements a classifier using a vocabulary consisting of bigrams
    """


if __name__ == "__main__":
    main()
