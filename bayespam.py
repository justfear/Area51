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


class Counter:

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


class Bayespam:

    def __init__(self):
        self.regular_list = None
        self.spam_list = None
        self.regular_results = []
        self.spam_results = []
        self.n_words_regular = 0
        self.n_words_spam = 0
        self.probability_regular = 0
        self.probability_spam = 0
        self.conditional_probabilities = defaultdict(dict)
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
        :param testing: Boolean for whether to read messages in a testing or training mindset (false by default)
        :return: None
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
                            ## Convert characters to lower case, remove punctuations, remove digits and new lines/breaks
                            token = "".join([char.lower() for char in token if char not in string.punctuation
                                             and not char.isdigit() and not re.search("[\\\\\\n\\t]", token)])
                            ## Encode to ascii and decode to utf-8 to remove hexadecimal numbers
                            token = token.encode('ascii', errors='ignore')
                            token = token.decode('utf-8')
                            ## Ensure we only add words with at least four letters
                            if len(token) >= 4:
                                ## Ensure the algorithm only learns when training.
                                ## Can allow it to learn while testing but would require supervision
                                ## Increase the count of regular words or spam words, depending on message type
                                if not testing:
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
                                        probability_regular += log10(self.conditional_probabilities[token]['regular'])
                                        probability_spam += log10(self.conditional_probabilities[token]['spam'])
                    f.close()
            except Exception as e:
                print("Error while reading message %s: " % msg, e)
                exit()
            ## Compare P(regular|msg) vs P(spam|msg)
            if testing:
                ## If P(spam|msg) > P(regular|msg) then we add the value True to the correct list depending on message type
                if message_type == MessageType.REGULAR:
                    self.regular_results.insert(msg_index, probability_regular < probability_spam)
                else:
                    self.spam_results.insert(msg_index, probability_regular < probability_spam)
            ## Increment msg_index by one to signal we move on to the next message
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
        :param sort_by_freq: Set to True to sort the vocab by total frequency (descending order)
        """

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
                        conditional_regular = 0.05 / (self.n_words_spam + self.n_words_regular)
                        conditional_spam = counter.counter_spam / self.n_words_spam
                    elif counter.counter_spam == 0:
                        conditional_regular = counter.counter_regular / self.n_words_regular
                        conditional_spam = 0.05 / (self.n_words_spam + self.n_words_regular)
                    ## Else compute the conditional probabilities normally
                    else:
                        conditional_regular = counter.counter_regular / self.n_words_regular
                        conditional_spam = counter.counter_spam / self.n_words_spam
                    ## Add the regular and spam conditional probabilities to the dict
                    self.conditional_probabilities[word]['regular'] = conditional_regular
                    self.conditional_probabilities[word]['spam'] = conditional_spam
                f.close()
        except Exception as e:
            print("An error occurred while writing the vocab to a file: ", e)
            exit()

    def train_data(self, train_path):
        """
        Computes and saves logP(regular), logP(spam) and all conditional probabilities P(Wj|regular), P(Wj|spam).

        :param train_path: The file path of the folder containing the train set from the input arguments
        :return: None
        """
        # Initialize a list of the regular and spam message locations in the training folder
        self.list_dirs(train_path)
        ## Compute probability of spam and regular messages
        self.compute_probabilities()
        # Parse the messages in the regular message directory,
        ##  and store the sum of total regular words
        self.read_messages(MessageType.REGULAR)
        # Parse the messages in the spam message directory,
        ## and store the sum of total spam words
        self.read_messages(MessageType.SPAM)
        ## Write each word and their occurrence in both spam and regular mail onto a text file,
        ##  and store the conditional probability that a word is in either spam or regular mail in a dict
        self.write_vocab(destination_fp="vocab.txt")


    def test_data(self, test_path):
        """
        Reads and parses each message in the designated regular and spam test folders,
        while having the test-mode parameter set to True.

        :param test_path The file path of the folder containing the test set from the input arguments
        :return: None
        """
        ## Initialize a list of the regular and spam message locations in the test folder
        self.list_dirs(test_path)
        ## Read and evaluate the regular test messages (testing mode: on)
        self.read_messages(MessageType.REGULAR, True)
        ## Read and evaluate the spam test messages (testing mode: on)
        self.read_messages(MessageType.SPAM, True)

    def confusion_matrix(self):
        """
        Counts each TP, TN, FP and FN values from the results lists.
        Prints a confusion matrix with those values.
        Also computes and prints the false accept, false reject and total accuracy rates.

        :return: None
        """
        ## We regard a true positive as the program correctly identifying a spam mail
        ## Count the number of False Positives (FPs)
        regular_t = self.regular_results.count(True)
        ## Count the number of True Negatives (TNs)
        regular_f = self.regular_results.count(False)
        ## Count the number of True Positives (TPs)
        spam_t = self.spam_results.count(True)
        ## Count the number of False Negatives (FNs)
        spam_f = self.spam_results.count(False)

        print("                   |               Actual Values                |")
        print("                   |  Positive (spam)         Negative (regular)|")
        print(" Predicted Values  |____________________________________________|")
        print(" Positive (spam)   | ", spam_t, "                    ", regular_t, "                |")
        print(" Negative (regular)| ", spam_f, "                    ", regular_f, "                |")
        print(" False Accept rate:", str(100 * spam_f / (regular_f + regular_t + spam_t + spam_f)) + "%  –– ",
              " False Reject rate:", str(100 * regular_t / (regular_f + regular_t + spam_t + spam_f)) + "%")
        print(" Total Accuracy rate: ", 100 * (regular_f + spam_t) / (regular_f + regular_t + spam_t + spam_f), "%")

    def compute_probabilities(self):
        """
        Computes and saves the log probabilities that a message is regular/spam.

        :return: None
        """
        ## Compute the number of regular mail messages
        n_messages_regular = len(self.regular_list)
        ## Compute the number of spam mail messages
        n_messages_spam = len(self.spam_list)
        ## Compute the total number of messages
        n_messages_total = n_messages_regular + n_messages_spam
        ## Compute the (log) probability that a message is regular logP(regular)
        print("Probability regular ", n_messages_regular / n_messages_total)
        self.probability_regular = log10(n_messages_regular / n_messages_total)
        ## Compute the (log) probability that a message is spam logP(spam)
        print("Probability spam ", n_messages_spam / n_messages_total)
        self.probability_spam = log10(n_messages_spam / n_messages_total)


def main():
    # We require the file paths of the training and test sets as input arguments (in that order)
    # The argparse library helps us cleanly parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', type=str,
                        help='File path of the directory containing the training data')
    parser.add_argument('test_path', type=str,
                        help='File path of the directory containing the test data')
    args = parser.parse_args()

    # Initialize a Bayespam object
    bayespam = Bayespam()
    ## Prompt the program in training itself on the training set
    bayespam.train_data(args.train_path)

    ## Prompt the program in classifying the testing set for both spam and regular messages
    bayespam.test_data(args.test_path)

    ## Create and print a confusion matrix
    bayespam.confusion_matrix()


if __name__ == "__main__":
    main()
