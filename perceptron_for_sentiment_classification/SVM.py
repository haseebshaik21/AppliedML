#!/usr/bin/env python3

import sys
import time
import numpy as np
from sklearn.svm import SVC

def read_from(textfile):
    data = []
    for line in open(textfile):
        label, words = line.strip().split("\t")
        data.append((1 if label == "+" else -1, words.split()))
    return data

def create_feature_vector(words, vocabulary):
    feature_vector = [1 if word in words else 0 for word in vocabulary]
    return feature_vector

def prune_words(data, min_count=1):
    word_counts = {}
    for _, words in data:
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

    pruned_data = [(label, [word for word in words if word_counts.get(word, 0) > min_count]) for label, words in data]
    return pruned_data

def build_vocabulary(data):
    vocabulary = set()
    for _, words in data:
        vocabulary.update(words)
    return list(vocabulary)

def train_svm(trainfile, devfile, num_samples=None):
    t = time.time()
    
    train_data = prune_words(read_from(trainfile))
    
    if num_samples is not None:
        train_data = train_data[:num_samples]

    dev_data = prune_words(read_from(devfile))
    
    vocabulary = build_vocabulary(train_data)
    
    X_train = [create_feature_vector(words, vocabulary) for _, words in train_data]
    y_train = [label for label, _ in train_data]
    
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)

    dev_err = test_svm(dev_data, clf, vocabulary)
    print("Development error: {:.2%}".format(dev_err))
    print("Time taken: {:.2f} seconds".format(time.time() - t))

def test_svm(data, clf, vocabulary):
    X_test = [create_feature_vector(words, vocabulary) for _, words in data]
    y_test = [label for label, _ in data]
    
    predictions = clf.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    return 1 - accuracy

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_svm.py <trainfile> <devfile>")
        sys.exit(1)

    trainfile, devfile = sys.argv[1], sys.argv[2]

    print("Training SVM with the following data:")
    print("Training Data:", trainfile)
    print("Development Data:", devfile)

    train_svm(trainfile, devfile, num_samples=5000)  # Specify the number of samples if needed
