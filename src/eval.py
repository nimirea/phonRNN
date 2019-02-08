import argparse
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.module import _addindent
import numpy as np
from collections import Counter
import csv
import os

import data

parser = argparse.ArgumentParser(description='PyTorch Phone-Level Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='data/processed/CELEX2/lemmas',
                    help='location of the data corpus')
parser.add_argument('--data_file', type=str, default='',
                    help='location of the test file (if unset, will test the whole corpus given in --data)')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--batch_dir', type=str, default='',
                    help='model checkpoints to use for batch operation (if unset, will only run script on one file)')
parser.add_argument('--summary_file', type=str, default='summary.csv',
                    help='file name of the summary file, within the batch_dir')
parser.add_argument('--out', type=str, default='final_eval-agg.csv',
                    help='name of output file to write the results to')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--stress', action='store_true',
                    help='keep track of word stress')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# print the model!
def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr

def get_batch(source, word_start, word_len, evaluation=False):
    seq_len = word_len - 1
    data = Variable(source[word_start:word_start+seq_len], volatile=evaluation)
    target = Variable(source[word_start+1:word_start+1+seq_len].transpose(1,0).contiguous().view(-1))
    return data, target

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def eval_test_file(checkpoint, corpus, test_file, include_stress, cuda):

    # process the test file using the corpus's dictionary
    test_data = corpus.tokenize(test_file)

    # load the model
    with open(checkpoint, 'rb') as f:
        model = torch.load(f)
    model.eval() # set to eval mode

    print(torch_summarize(model))

    if cuda:
        model.cuda()
    else:
        model.cpu()

    return(eval_data(model, corpus, test_data, include_stress, cuda, test_file))

def eval_data(model, corpus, tokenized_data, include_stress, cuda, data_id):
    # dictionary to store results
    out_dict = {}

    # basic characteristics of the corpus
    nchars = len(corpus.dictionary)

    # set model characteristics (common to all)
    criterion = nn.CrossEntropyLoss(size_average = False)
    total_loss = 0
    eval_batch_size = 1

    # initialize model
    hidden = model.init_hidden()

    # flatten
    tokenized_data['char_tensor'] = tokenized_data['char_tensor'].view(1,-1).t().contiguous()

    # move to GPU
    if cuda:
        tokenized_data['char_tensor'] = tokenized_data['char_tensor'].cuda()

    # loop through all words in this split
    current_word_start = 0
    idx_within_data = 0 # this is for keepign track of word results, to make sure duplicate words don't get overwritten
    for word_index, word_len in enumerate(tokenized_data['lengths']):
        data, targets = get_batch(tokenized_data['char_tensor'], current_word_start, word_len, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, nchars)
        this_loss = criterion(output_flat, targets).data

        # store by-word loss
        word = corpus.dictionary.translate(data.data[1:,0]) # don't include start symbol in this
        word_id = data_id + "_" + str(idx_within_data)
        out_dict[word_id] = {'loss': this_loss[0], 'word': word}

        hidden = repackage_hidden(hidden) # detach from history
        current_word_start += word_len
        idx_within_data += 1

    return(out_dict)


def eval_checkpoint(checkpoint, corpus, include_stress, cuda):
    out_dict = {} # variable that will store the results

    # load the model
    with open(checkpoint, 'rb') as f:
        model = torch.load(f)
    model.eval() # set to eval mode

    print(torch_summarize(model))

    if cuda:
        model.cuda()
    else:
        model.cpu()

    for split in corpus.splits:
        split_result = eval_data(model, corpus, split, include_stress, cuda, split['split_name'])

        # append result to dictionary
        out_dict = {**out_dict, **split_result}

    return out_dict

if args.batch_dir == "":
    print(eval_checkpoint(args.checkpoint, args.stress, args.cuda))
else:
    summary_filename = os.path.join(args.batch_dir, args.summary_file)
    out_filename = os.path.join(args.batch_dir, args.out)

    # set of checkpoints and various data about each
    checkpoint_info = {}

    with open(summary_filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row['save_dir'] in checkpoint_info:
                checkpoint_info[row['save_dir']] = {
                    'stress': row['stress'] == "True",
                    'cuda': row['cuda'] == "True",
                    'condition': row['condition'],
                    'run': row['run'],
                    'save_dir': row['save_dir']
                }

    # perform all calculations

    words = {}
    evaluations = {}

    # initialize corpus
    corpus = data.Corpus(args.data, stress=args.stress)

    # initialize the dictionary of words to test
    if args.data_file == '':
        for split in corpus.splits:
            words_in_split = 0
            duped = 0
            for w, word in enumerate(split['plain_words']):
                word_id = split['split_name'] + "_" + str(w)
                words[word_id] = {
                    'word': word, # redundant but will help later
                    'len': split['lengths'][w] - 2, # don't include start or end symbol in length
                    'split_name': split['split_name']
                }
    else:
        with open(args.data_file, 'r') as test_file:
            idx = 0
            for line in test_file:
                word = line[:-1]
                word_id = args.data_file + "_" + str(w)
                words[word_id] = {
                    'word': word,
                    'len': len(word.split(" ")),
                    'split_name': args.data_file
                }
                idx += 1

    field_names = ['word', 'len', 'split_name']

    # loop through checkpoint files
    for checkpoint_dir, checkpoint_data in checkpoint_info.items():

        # get best model file
        checkpoint_file = os.path.join(checkpoint_dir, "best-model.pt")

        # calculate evaluation statistics
        if args.data_file == '':
            eval_dict = eval_checkpoint(checkpoint_file, corpus, checkpoint_data['stress'], checkpoint_data['cuda'])
        else:
            eval_dict = eval_test_file(checkpoint_file, corpus, args.data_file, checkpoint_data['stress'], checkpoint_data['cuda'])

        checkpoint_index = os.path.basename(checkpoint_dir)
        # store evaluation statistics
        words_in_evaldict = 0
        for word_id, word_data in eval_dict.items():
            words_in_evaldict += 1
            words[word_id][checkpoint_index] = word_data['loss']
        print("total words in eval_dict: " + str(words_in_evaldict))

        field_names.append(checkpoint_index)

    with open(out_filename, 'w') as csvfile:
        # initialize with correct filenames
        writer = csv.DictWriter(csvfile, fieldnames = field_names)
        writer.writeheader()

        for word, eval_dict in words.items():
            print(eval_dict)
            writer.writerow(eval_dict)
