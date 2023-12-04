#!/usr/bin/env python3

from __future__ import division

import sys
import time
from svector import svector

def read_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label == "+" else -1, words.split())

def make_vector(words):
    v = svector()
    for word in words:
        v[word] += 1
    v["bias"] = 1
    return v

def test(devfile, model):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1):
        err += label * (model.dot(make_vector(words))) <= 0
    return err / i

def train(trainfile, devfile, epochs=5):
    t = time.time()
    best_err = 1.0
    model = svector()
    total_updates = 0
    avg_model = svector()

    word_counts = {}

    for _, words in read_from(trainfile):
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

    for it in range(1, epochs + 1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1):
            filtered_words = [word for word in words if word_counts.get(word, 0) > 1]
            sent = make_vector(filtered_words)
            if label * model.dot(sent) <= 0:
                updates += 1
                model += label * sent
            total_updates += updates
            avg_model += total_updates * model

        dev_err = test(devfile, avg_model / total_updates)
        best_err = min(best_err, dev_err)
        print(f"epoch {it}, update {updates / i * 100:.1f}%, dev {dev_err * 100:.1f}%")

    print(f"best dev err {best_err * 100:.1f}%, |w|={len(avg_model)}, time: {time.time() - t:.1f} secs")

def predict_and_save(testfile, model, output_file):
    with open(output_file, "w") as outfile:
        for _, words in read_from(testfile):
            sent = make_vector(words)
            prediction = model.dot(sent)
            label = "+" if prediction > 0 else "-"
            outfile.write(f"{label}\t{' '.join(words)}\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 train.py train.txt dev.txt test.txt")
        sys.exit(1)
    train(sys.argv[1], sys.argv[2], 10)
