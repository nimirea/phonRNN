#!/usr/bin/env python

from __future__ import print_function

# Use the srilm module
import srilm

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--out", help="path to output file")
parser.add_argument("--lm", help="path to language model file")
parser.add_argument("--test", help="path to test file")
args = parser.parse_args()

def test_lm(lm_filename, test_filename, max_n = 9):
    '''
    - lm_filename = filename of the language model to test
    - test_filename = filename of test set
    - max_n = maximum n to test in ngram
    '''

    # build word dictionary
    words = {}
    # initialize lengths dictionary
    lengths = {}
    with open(test_filename, 'r') as f:
        for word in f:
            start_size = len(words)

            # strip out whitespace and beginning of line characters
            orig_word = word
            word = word.rstrip()
            word = word.replace("<s> ", "")

            # add to dictionaries
            words[word] = []
            lengths[word] = len(word.split(" "))

    # get probability for each word
    for n in range(1, max_n+1):
        ngram_model = srilm.initLM(n)
        srilm.readLM(ngram_model, lm_filename)

        for word in words:
            wordprob = srilm.getSentenceProb(ngram_model, word, lengths[word])
            words[word].append(wordprob)

    output = "word,len,"
    output += ",".join([str(ng) + "gram" for ng in range(1, max_n+1)])

    for word, probs in words.iteritems():
        output += "\n" + word + ",{},{}".format(lengths[word], ",".join([str(p) for p in probs]))

    return output

def get_problist(corpus_path, lm_var, n):
    wordprobs = []
    with open(corpus_path, 'r') as r:
        word = f.readline()
        word = word[4:] # strip start character
        wordprob = srilm.getSentenceProb(lm_var, word, n)

# outfile = 'ngram_results_stress.csv'
# lmfile = '9gram_WOLEX_stress.txt'
# testfile = '../../pytorch/data/WOLEX_nodiph/testval_combined.txt'

output = test_lm(args.lm, args.test)
with open(args.out, 'w') as out:
    out.write(output)
