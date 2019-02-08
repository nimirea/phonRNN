import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import pronouncing
from operator import itemgetter
from random import randint, shuffle
import features
import numpy as np

class Dictionary(object):
    def __init__(self):
        self.phone2idx = {}
        self.idx2phone = []

        # add padding character to dictionary
        self.padding_char = '.'
        self.add_phone(self.padding_char)

    def add_phone(self, phone):
        if phone not in self.phone2idx:
            self.idx2phone.append(phone)
            self.phone2idx[phone] = len(self.idx2phone) - 1
        return self.phone2idx[phone]

    """
    Return feature embedding
    """
    def feature_embedding(self, encoding = "IPA", ftree = "Futrell", set_unset = False):
        # which encoding is the data in?
        if encoding == "ARPA":
            parser = features.ARPArser()
        elif encoding == "IPA":
            parser = features.IPArser()

        # which feature tree should we be using?
        if ftree == "Futrell":
            feature_tree = features.FutrellFeatures(parser, set_unset = set_unset)
        elif ftree == "orig":
            feature_tree = features.OrigFeatures(parser, set_unset = set_unset)

        feature_vec_len = len(feature_tree.feature_list) + len(feature_tree.feature_list) * set_unset
        embeddings = torch.FloatTensor(len(self.idx2phone), feature_vec_len)
        for phone_idx, phone in enumerate(self.idx2phone):
            embeddings[phone_idx] = feature_tree.to_feature_vec(phone)

        rows, cols = embeddings.shape
        embedding = nn.Embedding(num_embeddings = rows, embedding_dim = cols)
        embedding.weight = nn.Parameter(embeddings)

        return embedding

    """
    Return random embedding
    """
    def random_embedding(self, encoding = "IPA", ftree = "Futrell", set_unset = False):
        # generate feature embedding
        feat_emb = self.feature_embedding(encoding, ftree, set_unset).weight.data
        vals, counts = np.unique(feat_emb.numpy(), return_counts = True)
        probs = counts / sum(counts)

        rows, cols = feat_emb.shape

        dist_weights = np.random.choice(vals, (rows, cols), replace=True, p=probs)
        embeddings = torch.FloatTensor(dist_weights)

        embedding = nn.Embedding(num_embeddings = rows, embedding_dim = cols)
        embedding.weight = nn.Parameter(embeddings)

        return embedding

    def translate(self, tensor):
        return " ".join([self.idx2phone[t] for t in tensor])

    def __len__(self):
        return len(self.idx2phone)

class RawCorpus(object):
    # class to deal with the switchboard corpus
    def __init__(self, path, ext = "txt"):
        self.path = path
        self.transcripts = []
        print(os.path.abspath(path))

        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith("." + ext):
                    self.transcripts += [os.path.join(root,file)]

    def word_types_freqranked(self):
        freq_dict = {}
        word_types = 0

        for transcript in self.transcripts:
            with open(transcript, 'r') as f:
                for line in f:
                    for word in line.split():
                        if word in freq_dict:
                            freq_dict[word] += 1
                        else:
                            freq_dict[word] = 1
                            word_types += 1

        freq_tuples = list(freq_dict.items())
        freq_tuples.sort(key=itemgetter(1), reverse=True)
        words_sorted = [tup[0] for tup in freq_tuples]

        return words_sorted

    def cmu_transcribe(self, word_list):
        """
        Transcribe a list of words using the CMU pronouncing dictionary.

        INPUT:
        - list of word spellings

        RETURNS
        - prons: list of CMU pronunciations for each word in the word_list
        - unk_words: list of spellings of words not found in CMU pronouncing dictionary
        """
        unk_words = []
        prons = []
        known_words = set()

        for raw_word in word_list:
            word = raw_word.lower()
            entry = pronouncing.phones_for_word(word)

            if entry:

                # add random pronunciation to the list of forms pronounced
                pron = entry[randint(0,len(entry)-1)]
                prons.append(pron)

                known_words.add(word)

            else: # no pronunciation found in dictionary
                unk_words.append(raw_word)

        return prons, unk_words

    def top_prons(self, n_prons):
        """Get the CMU pronunciations of the most frequent n_prons words in this corpus."""
        words_sorted = self.word_types_freqranked()
        all_prons, unks = self.cmu_transcribe(words_sorted)
        top_prons = all_prons[0:n_prons]

        print("{} unique words in the vocabulary".format(len(words_sorted)))
        print("{}/{} pronunciations found".format(len(top_prons), n_prons))
        print("Unknown words: {}".format(len(unks)))

        return top_prons

    def write_splits(self, output_dir = None, n_prons = 6250, val_frac = 0.1, test_frac = 0.1, padded = True, stress = True):
        if output_dir == None:
            output_dir = os.path.join(self.path, 'processed')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prons = self.top_prons(n_prons)

        # shuffle pronunciations
        shuffle(prons)

        # calculate split sizes
        val_words = int(val_frac * n_prons)
        test_words = int(test_frac * n_prons)
        train_words = n_prons - val_words - test_words

        val_chars = prons[:val_words]
        test_chars = prons[val_words:val_words + test_words]
        train_chars = prons[-train_words:]
        splits = {
            'train': train_chars,
            'valid': val_chars,
            'test': test_chars
        }

        # split sizes, in phones
        val_size = sum([len(x) for x in val_chars])
        test_size = sum([len(x) for x in test_chars])
        train_size = sum([len(x) for x in train_chars])

        # write splits to files
        for split_name, split_words in splits.items():
            out_file = os.path.join(output_dir, split_name + '.txt')
            with open(out_file, 'w') as out:
                for word in split_words:
                    out.write("%s\n" % "".join(word))

        return output_dir

    def process(self, output_dir = None, n_prons = 6250, val_frac = 0.1, test_frac = 0.1, padded = True):
        if output_dir == None:
            output_dir = os.path.join(self.path, 'processed')
        processed_path = self.write_splits(output_dir, n_prons, val_frac, test_frac = 0.1, padded = True)
        return Corpus(processed_path)

class Corpus(object):
    def __init__(self, path, stress=True):
        self.dictionary = Dictionary()
        self.root = path
        self.stress = stress
        self.longest_length, self.n_words_in = self.stats()

        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

        self.splits = [self.train, self.valid, self.test]

    def stats(self):
        """Find number of phones in the longest word in the corpus, along with the number of words in each split."""

        # initialize names of the splits
        split_names = ['test', 'train', 'valid']
        paths = [os.path.join(self.root, x + '.txt') for x in split_names]

        # initialize return variables
        longest_length = 0
        n_words_in = {}

        for p, path in enumerate(paths):
            n_words_in[split_names[p]] = 0
            with open(path, 'r') as f:
                for line in f:
                    phones = line.split()
                    n_words_in[split_names[p]] += 1
                    if len(phones) > longest_length:
                        longest_length = len(phones)

        # add 1 for the start symbol "end of line" character
        return longest_length + 2, n_words_in

    def write_splits(self, out_path, split_by_length = False):
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        # this is kind of ham-handed but whatever
        self.write_split(self.train, os.path.join(out_path, 'train'), split_by_length)
        self.write_split(self.test, os.path.join(out_path, 'test'), split_by_length)
        self.write_split(self.valid, os.path.join(out_path, 'valid'), split_by_length)

    def write_split(self, split, filename, split_by_length = False):
        if not split_by_length:
            filename += '.txt'
            with open(filename, 'w') as outfile:
                for char_id in split['char_tensor']:
                    char_to_write = self.dictionary.idx2phone[char_id]
                    if char_to_write != "\n":
                        char_to_write += " "
                    outfile.write(char_to_write)
        else:
            output_dir = filename
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            current_idx = 0
            for word_len in split['lengths']:
                true_word_len = word_len - 1
                out_file = os.path.join(output_dir, str(true_word_len) + '.txt')
                with open(out_file, 'a') as out:
                    for char_id in split['char_tensor'][current_idx:current_idx+word_len]:
                        char_to_write = self.dictionary.idx2phone[char_id]
                        if char_to_write != "\n":
                            char_to_write += " "
                        out.write(char_to_write)
                current_idx += word_len

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        n_words = 0
        n_tokens = 0
        lengths = []
        words = []
        with open(path, 'r') as f:
            for line in f:
                # strip stress, if applicable
                if not self.stress:
                    raw_line = ''.join([i for i in line if not i.isdigit()])
                else:
                    raw_line = line

                word = raw_line.strip().split(" ")

                # add start and end symbol (newline) to word, if not there already
                if word[0] != '<s>':
                    word = ['<s>'] + word
                word = word + ['\n']

                words.append(word)
                n_words += 1
                lengths.append(len(word))
                for phone in word:
                    n_tokens += 1
                    self.dictionary.add_phone(phone)

        ids = torch.LongTensor(n_tokens)
        token = 0
        plain_words = [] # store words without start/end symbols
        for word in words:
            for phone in word:
                ids[token] = self.dictionary.phone2idx[phone]
                token += 1
            plain_words.append(" ".join(word[1:-1]))

        split_name = os.path.splitext(os.path.basename(path))[0]

        return {'char_tensor': ids,
                'lengths': lengths,
                'split_name': split_name,
                'plain_words': plain_words}
